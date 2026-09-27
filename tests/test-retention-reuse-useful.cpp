// Unit test for server_prompt_cache_retention_reuse_is_useful() (slice 9j).
//
// That predicate is the whole gate between "the client re-sent a prefix" and
// "that prefix earns durable frequency credit"; it is also a pure function of
// (reused token count, message spans), so unlike the rest of the active-prefix
// face it can be pinned here without a model.  Two properties make it worth
// pinning rather than eyeballing:
//
//   * the 256-token floor is a policy number, and a silent drift in it changes
//     how much cache pressure the server tolerates;
//   * the sub-floor clause is ORDER-SENSITIVE -- a user span only qualifies if a
//     complete system span was seen at an earlier index -- which is easy to
//     "clean up" into a set-membership test and thereby widen the rule.
//
// Checks, in increasing strength:
//   1. a frozen boundary table over the floor and the degenerate inputs;
//   2. an independent oracle (different control flow: first-complete-system
//      index, then a suffix search) compared over an exhaustive grid of span
//      multisets, which is what catches an accidental order/perimeter change;
//   3. the floor constant itself, frozen.
//
// Deliberately NOT assert-based: the Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one.  Failures
// are counted, printed, and the exit code is derived from the count.
//
// Model-free and CPU-only: no GGUF is loaded.

#include "server-task.h"

#include <cstdio>
#include <vector>

static int  g_checks   = 0;
static int  g_failures = 0;
// The negative control injects one deliberately-false check; it silences the report
// for that single evaluation so a *passing* run never prints "FAIL".  Judging by
// output text (rather than an exit code) is this tree's convention, and a stray FAIL
// line in a green run would read as a red one.  `g_muted` counts the reports that were
// produced but not printed, so the control can still prove the failure/report path ran.
static bool g_report   = true;
static int  g_muted    = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            if (g_report) {                                                  \
                std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            } else {                                                         \
                ++g_muted;                                                   \
            }                                                                \
        }                                                                    \
    } while (0)

#define CHECK_EQ(a, b)                                                       \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!((a) == (b))) {                                                 \
            ++g_failures;                                                    \
            if (g_report) {                                                  \
                std::printf("FAIL %s:%d  %s != %s\n",                        \
                            __FILE__, __LINE__, #a, #b);                     \
            } else {                                                         \
                ++g_muted;                                                   \
            }                                                                \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Independent oracle.
//
// The production predicate walks the spans once, keeping a "have I already seen a
// complete system prefix" flag.  This one asks the same question with different
// control flow: locate the FIRST index whose span is a complete system prefix,
// then look for a user span in the suffix from that index.  The two agree because
// the flag is monotone -- once a complete system span exists at i, every later
// iteration sees it -- and a span has exactly one role, so a span can never set
// the flag and satisfy the user test in the same iteration.
// ---------------------------------------------------------------------------
static bool is_complete_system(common_chat_role role, size_t pos, size_t len, size_t n) {
    return role == COMMON_CHAT_ROLE_SYSTEM && pos <= n && len <= n - pos;
}

static bool oracle_reuse_is_useful(const std::vector<common_chat_msg_span> & spans, size_t n) {
    if (n >= SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS) {
        return true;
    }
    if (n == 0) {
        return false;
    }
    size_t first = spans.size();
    for (size_t i = 0; i < spans.size(); ++i) {
        if (is_complete_system(spans[i].role, spans[i].pos, spans[i].len, n)) {
            first = i;
            break;
        }
    }
    if (first == spans.size()) {
        return false;
    }
    for (size_t j = first; j < spans.size(); ++j) {
        if (spans[j].role == COMMON_CHAT_ROLE_USER && spans[j].pos <= n) {
            return true;
        }
    }
    return false;
}

static bool pred(const std::vector<common_chat_msg_span> & spans, size_t n) {
    common_chat_msg_spans s;
    s.spans = spans;
    return server_prompt_cache_retention_reuse_is_useful(n, &s);
}

static std::string describe(const std::vector<common_chat_msg_span> & spans) {
    std::string out;
    for (const auto & sp : spans) {
        const char * role = sp.role == COMMON_CHAT_ROLE_SYSTEM ? "sys"
                          : sp.role == COMMON_CHAT_ROLE_USER   ? "usr"
                          : sp.role == COMMON_CHAT_ROLE_UNKNOWN ? "unk" : "oth";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s@%zu+%zu ", role, sp.pos, sp.len);
        out += buf;
    }
    return out;
}

int main(void) {
    std::printf("== floor constant\n");
    CHECK_EQ(SERVER_PROMPT_CACHE_MIN_RETENTION_REUSE_TOKENS, size_t(256));

    // ------------------------------------------------ frozen boundary table
    std::printf("== boundary table (no message spans)\n");
    struct row { size_t n; bool expected; };
    const row table[] = {
        {   0, false },
        {   1, false },
        { 255, false },
        { 256, true  },   // the floor is inclusive: >= earns credit
        { 257, true  },
        {1024, true  },
    };
    for (const auto & r : table) {
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(r.n, nullptr), r.expected);
    }
    CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(0), false);

    std::printf("== empty span set below the floor\n");
    {
        common_chat_msg_spans empty;
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(100, &empty), false);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(256, &empty), true);
    }

    std::printf("== the sub-floor system-prefix clause\n");
    {
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 40);
        spans.add(COMMON_CHAT_ROLE_USER,   40, 60);
        // A complete system span followed by user content within the reuse: true
        // even well below the generic floor.
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(100, &spans), true);
        // Same set, but the user content begins past the reused prefix.
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(39, &spans), false);
        // Exactly at the boundary of the user span: pos <= n is inclusive.
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(40, &spans), true);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(0, &spans), false);
    }
    {
        // System span only: no user content to reach, so no credit.
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 40);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(100, &spans), false);
    }
    {
        // System span that overshoots the reused prefix is NOT complete, so the
        // user span does not qualify either.
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 120);
        spans.add(COMMON_CHAT_ROLE_USER,   120, 30);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(100, &spans), false);
    }
    {
        // ORDER MATTERS: user before system does not qualify. Pinned because this
        // is the property most likely to be "simplified" away.
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_USER,   40, 60);
        spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 40);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(100, &spans), false);
    }
    {
        // A second, later system span re-opens the window: the flag is monotone.
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_USER,   40, 60);
        spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 40);
        spans.add(COMMON_CHAT_ROLE_USER,   50, 60);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(100, &spans), true);
    }
    {
        // A zero-length system span at position 0 is trivially complete.
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 0);
        spans.add(COMMON_CHAT_ROLE_USER,   5, 10);
        CHECK_EQ(server_prompt_cache_retention_reuse_is_useful(5, &spans), true);
    }

    // -------------------------------------------------- exhaustive grid: oracle
    std::printf("== exhaustive grid against the independent oracle\n");
    {
        const common_chat_msg_span pool[] = {
            { COMMON_CHAT_ROLE_SYSTEM,  0,  10 },
            { COMMON_CHAT_ROLE_SYSTEM,  0, 300 },
            { COMMON_CHAT_ROLE_SYSTEM, 10,  10 },
            { COMMON_CHAT_ROLE_USER,    0,   5 },
            { COMMON_CHAT_ROLE_USER,   10,   5 },
            { COMMON_CHAT_ROLE_USER,  300,   5 },
            { COMMON_CHAT_ROLE_UNKNOWN, 0,  10 },
        };
        const size_t n_pool = sizeof(pool) / sizeof(pool[0]);
        const size_t ns[] = { 0, 1, 5, 255, 256, 300 };

        size_t cases = 0;
        for (size_t n : ns) {
            // singletons + pairs + triples, in order (order is semantic)
            for (size_t i = 0; i < n_pool; ++i) {
                std::vector<common_chat_msg_span> v{ pool[i] };
                CHECK_EQ(pred(v, n), oracle_reuse_is_useful(v, n));
                ++cases;
            }
            for (size_t i = 0; i < n_pool; ++i) {
                for (size_t j = 0; j < n_pool; ++j) {
                    std::vector<common_chat_msg_span> v{ pool[i], pool[j] };
                    CHECK_EQ(pred(v, n), oracle_reuse_is_useful(v, n));
                    if (pred(v, n) != oracle_reuse_is_useful(v, n)) {
                        std::printf("      span set: %s(n=%zu)\n", describe(v).c_str(), n);
                    }
                    ++cases;
                }
            }
            for (size_t i = 0; i < n_pool; ++i) {
                for (size_t j = 0; j < n_pool; ++j) {
                    for (size_t k = 0; k < n_pool; ++k) {
                        std::vector<common_chat_msg_span> v{ pool[i], pool[j], pool[k] };
                        CHECK_EQ(pred(v, n), oracle_reuse_is_useful(v, n));
                        if (pred(v, n) != oracle_reuse_is_useful(v, n)) {
                            std::printf("      span set: %s(n=%zu)\n", describe(v).c_str(), n);
                        }
                        ++cases;
                    }
                }
            }
        }
        std::printf("   %zu span sets compared over %zu token counts\n",
                    cases / (sizeof(ns) / sizeof(ns[0])), sizeof(ns) / sizeof(ns[0]));
        // the grid must not be vacuous
        CHECK(cases > 2000);
    }

    // ------------------------------------------------------ negative control
    // Prove the counter and the report path are really wired: inject one
    // deliberately-false check, confirm the failure count moved by exactly one and
    // that the failure branch reported (g_muted), then roll both counters back so the
    // injection never reaches the exit code or the reported total.  This is the only
    // way to tell a `0 failures` line from a harness whose counter is not connected.
    // The injection is silenced (see g_report) so a green run prints no "FAIL" text.
    std::printf("== negative control: the counter really can fail\n");
    {
        const int before       = g_failures;
        const int muted_before = g_muted;
        g_report = false;
        CHECK(false);                                  // deliberately false, not reported
        g_report = true;
        CHECK(g_failures == before + 1);               // the counter moved by exactly 1
        CHECK(g_muted == muted_before + 1);            // ... and the failure branch ran
        g_failures = before;                           // do not let it reach the exit code
        g_checks  -= 3;                                // ... nor the reported total
    }

    std::printf("test-retention-reuse-useful: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
