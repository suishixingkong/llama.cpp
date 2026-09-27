// Unit test for the cache-receipt chain landed with the --cache-receipt feature
// (slice 9i). `cache_receipt_chain` is the only live part of that feature: it is
// a pure function over (tokens, block size, key), so it can be pinned here
// without a model, and pinning it matters because a receipt is a *comparison
// hint* -- if the chain drifts silently, two clients that should agree will
// report divergence.
//
// Three independent checks, in increasing strength:
//   1. shape and boundaries (block size 0, block >= n, empty input)
//   2. a second SHA-256 in this tree (src/llama-sha256.h) recomputes every
//      digest, so a transcription error inside the newly landed local SHA-256
//      cannot hide behind itself
//   3. frozen golden digests produced by a *third* implementation (Python
//      hashlib), which is the only one of the three that shares no code
//
// Deliberately NOT assert-based: the Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one. Failures
// are counted, printed, and the exit code is derived from the count.
//
// Model-free and CPU-only: no GGUF is loaded.

#include "server-common.h"

#include "../src/llama-sha256.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

static int g_checks   = 0;
static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
        }                                                                    \
    } while (0)

#define CHECK_EQ(a, b)                                                       \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!((a) == (b))) {                                                 \
            ++g_failures;                                                    \
            std::printf("FAIL %s:%d  %s != %s\n",                            \
                        __FILE__, __LINE__, #a, #b);                         \
        }                                                                    \
    } while (0)

static std::string join(const std::vector<std::string> & v) {
    std::string s;
    for (const auto & e : v) {
        if (!s.empty()) {
            s += ",";
        }
        s += e;
    }
    return s;
}

// Independent recomputation: same algorithm, different SHA-256 implementation.
static std::vector<std::string> oracle_chain(
        const std::vector<llama_token> & tokens,
        uint32_t                          block_tokens,
        const std::string &               key) {
    std::vector<std::string> chain;
    if (block_tokens == 0) {
        return chain;
    }
    std::array<uint8_t, 32> prev{};
    static const char hexd[] = "0123456789abcdef";
    for (size_t off = 0; off < tokens.size(); off += block_tokens) {
        const size_t n = std::min<size_t>(block_tokens, tokens.size() - off);
        llama_sha256 h;
        h.update(key.data(), key.size());
        h.update(prev.data(), prev.size());
        h.update(tokens.data() + off, n * sizeof(llama_token));
        prev = h.finish();
        std::string hex;
        for (int i = 0; i < 8; ++i) {
            hex.push_back(hexd[prev[i] >> 4]);
            hex.push_back(hexd[prev[i] & 0xf]);
        }
        chain.push_back(std::move(hex));
    }
    return chain;
}

// ---------------------------------------------------------------- the tests

static void test_shape_and_boundaries() {
    std::printf("== shape and boundaries\n");
    const std::vector<llama_token> four = {1, 2, 3, 4};

    // block_tokens == 0 is the documented refusal: no chain at all, not one
    // oversized block
    CHECK(cache_receipt_chain(four, 0, "k").empty());
    CHECK(cache_receipt_chain({}, 1024, "k").empty());

    // ceil(n / block) blocks
    CHECK_EQ(cache_receipt_chain(four, 1, "k").size(), size_t(4));
    CHECK_EQ(cache_receipt_chain(four, 2, "k").size(), size_t(2));
    CHECK_EQ(cache_receipt_chain(four, 3, "k").size(), size_t(2));
    CHECK_EQ(cache_receipt_chain(four, 4, "k").size(), size_t(1));
    CHECK_EQ(cache_receipt_chain(four, 1024, "k").size(), size_t(1));

    // every entry is trunc64 rendered as 16 lowercase hex digits
    for (const auto & d : cache_receipt_chain(four, 1, "k")) {
        CHECK_EQ(d.size(), size_t(16));
        CHECK(d.find_first_not_of("0123456789abcdef") == std::string::npos);
    }

    // deterministic
    CHECK_EQ(cache_receipt_chain(four, 2, "k"), cache_receipt_chain(four, 2, "k"));
}

static void test_oracle_agrees() {
    std::printf("== a second SHA-256 in this tree agrees on every digest\n");
    const std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    const uint32_t blocks[] = {1, 2, 3, 4, 5, 1024};
    const char *   keys[]   = {"tenant-key", "", "another", "tenant-key"};
    for (uint32_t b : blocks) {
        for (const char * k : keys) {
            const auto got = cache_receipt_chain(tokens, b, k);
            const auto want = oracle_chain(tokens, b, k);
            CHECK(got == want);
            if (got != want) {
                std::printf("     block=%u key=%s\n     got  %s\n     want %s\n",
                            b, k, join(got).c_str(), join(want).c_str());
            }
        }
    }
    // and on inputs that straddle the 64-byte SHA-256 block boundary, where a
    // buffer-management bug in either implementation would show up
    for (size_t n = 0; n <= 40; ++n) {
        std::vector<llama_token> t(n);
        for (size_t i = 0; i < n; ++i) {
            t[i] = llama_token(1000 + i * 7);
        }
        CHECK(cache_receipt_chain(t, 3, "straddle") == oracle_chain(t, 3, "straddle"));
    }
}

static void test_golden() {
    std::printf("== frozen golden digests (produced by an unrelated implementation)\n");
    // tokens {1,2,3,4,5}, block 2, key "tenant-key":
    //   h_i = trunc64(SHA256(key || h_{i-1} || block_i)), h_{-1} = 0^32
    const std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    const std::vector<std::string> want_keyed = {
        "8bb0eced49e7ecaa", "01d9270bee576a39", "ce2e15ba0a591a5f",
    };
    const std::vector<std::string> want_unkeyed = {
        "e7ca17d6581d126a", "a9edfe541f21e65d", "d0959160a34210cd",
    };
    CHECK_EQ(cache_receipt_chain(tokens, 2, "tenant-key"), want_keyed);
    CHECK_EQ(cache_receipt_chain(tokens, 2, ""),           want_unkeyed);

    // a single whole-prompt block (the block size the server actually uses)
    const std::vector<std::string> want_1024 = {"bc40045abc7701bd"};
    CHECK_EQ(cache_receipt_chain({7, 8}, 1024, "k"), want_1024);
}

static void test_key_sensitivity() {
    std::printf("== the key changes the chain; unkeyed is not a prefix of keyed\n");
    const std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    const auto a = cache_receipt_chain(tokens, 2, "tenant-a");
    const auto b = cache_receipt_chain(tokens, 2, "tenant-b");
    const auto u = cache_receipt_chain(tokens, 2, "");
    CHECK(a != b);
    CHECK(a != u);
    CHECK_EQ(a.size(), b.size());
    // the key is mixed into *every* block, so a key change must move even the
    // last digest, not merely the first
    CHECK(a[0] != b[0]);
    CHECK(a[2] != b[2]);
}

static void test_chaining() {
    std::printf("== each digest depends on its predecessor and nothing later\n");
    // changing a token inside the FIRST block must move every later digest
    const auto base = cache_receipt_chain({1, 2, 3, 4, 5, 6}, 2, "k");
    const auto first_moved = cache_receipt_chain({1, 9, 3, 4, 5, 6}, 2, "k");
    CHECK_EQ(base.size(), size_t(3));
    CHECK(base[0] != first_moved[0]);
    CHECK(base[1] != first_moved[1]);
    CHECK(base[2] != first_moved[2]);

    // changing a token inside the LAST block must leave the earlier digests
    // untouched -- this is what makes the receipt a *divergence locator*: the
    // first differing block is where the two sequences part
    const auto last_moved = cache_receipt_chain({1, 2, 3, 4, 5, 9}, 2, "k");
    CHECK_EQ(base[0], last_moved[0]);
    CHECK_EQ(base[1], last_moved[1]);
    CHECK(base[2] != last_moved[2]);
}

static void test_prefix_property() {
    std::printf("== a shorter prompt's chain is a prefix of a longer one's\n");
    const auto short_ch = cache_receipt_chain({1, 2, 3, 4}, 2, "tenant-key");
    const auto long_ch  = cache_receipt_chain({1, 2, 3, 4, 5, 6}, 2, "tenant-key");
    CHECK_EQ(short_ch.size(), size_t(2));
    CHECK_EQ(long_ch.size(), size_t(3));
    CHECK_EQ(short_ch[0], long_ch[0]);
    CHECK_EQ(short_ch[1], long_ch[1]);

    // ... but only while the block grid is aligned: offsetting the same tokens
    // by one must change the first digest, because the block contents differ
    const auto offset = cache_receipt_chain({0, 1, 2, 3}, 2, "tenant-key");
    CHECK(offset[0] != short_ch[0]);
}

static void test_payload_model_defaults() {
    std::printf("== the receipt payload model's defaults\n");
    // The struct has no runtime consumer (the server builds the same JSON
    // inline), so its defaults are pinned here: they document the wire shape
    // and a wrong default would silently mislabel every emitted receipt.
    const server_cache_receipt r;
    CHECK_EQ(r.version,         uint32_t(1));
    CHECK_EQ(r.block_tokens,    uint32_t(1024));
    CHECK_EQ(r.sequence_epoch,  uint64_t(0));
    CHECK_EQ(r.token_count,     int64_t(0));
    CHECK(r.hash_spec.empty());
    CHECK(r.model_identity.empty());
    CHECK(r.adapter_identity.empty());
    CHECK(r.chain.empty());
}

int main() {
    test_shape_and_boundaries();
    test_oracle_agrees();
    test_golden();
    test_key_sensitivity();
    test_chaining();
    test_prefix_property();
    test_payload_model_defaults();

    // ------------------------------------------------------ negative control
    std::printf("== negative control: the counter really can fail\n");
    {
        const int before = g_failures;
        CHECK(false);                                  // deliberately false
        CHECK(g_failures == before + 1);               // the counter moved by 1
        g_failures = before;                           // do not let it reach the exit code
        g_checks  -= 2;                                // ... nor the reported total
    }

    std::printf("test-cache-receipt-chain: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
