// Unit test for the MTP carry lifecycle and its host-checkpoint codec (slice 9h).
//
// common_speculative_mtp_carry_lifecycle::target_process_mode() decides, for a
// nonzero frontier, whether the target may consume a retained hidden row or
// must run target-only -- and getting that backwards makes a restored sequence
// silently read the previous branch's row. state_save/state_load are the codec
// that row travels through. Both are pure logic over a bool and a float vector:
// no model, no context, so this test pins the truth table and the wire format
// against transcription errors.
//
// Deliberately NOT assert-based: the Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one. Failures
// are counted, the actual check count is printed, and the exit code derives
// from the count. A negative control at the end proves the counter can fail.
//
// Model-free and CPU-only: no GGUF is loaded.

#include "speculative.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_checks = 0;
static int g_fails  = 0;

static void check(bool ok, const char * what) {
    g_checks++;
    if (!ok) {
        g_fails++;
        printf("  FAIL %s\n", what);
    }
}

using lifecycle = common_speculative_mtp_carry_lifecycle;
using pmode     = lifecycle::process_mode;

// every event a restored/cloned sequence can raise
static const common_speculative_sequence_event k_events[] = {
    common_speculative_sequence_event::prompt_rewind,
    common_speculative_sequence_event::target_restored_without_draft,
    common_speculative_sequence_event::draft_image_restored,
    common_speculative_sequence_event::composite_image_restored,
    common_speculative_sequence_event::live_range_shift,
    common_speculative_sequence_event::target_replaced,
    common_speculative_sequence_event::full_clear,
};

int main(void) {
    printf("test-speculative-mtp-carry: lifecycle truth table + codec wire format\n");

    // ---------------------------------------------------------------- cold
    printf("== a cold lifecycle carries nothing ==\n");
    {
        lifecycle lc;
        const float pending[2] = { 1.0f, 2.0f };
        check(!lc.draft_ready(),                      "cold: draft_ready() is false");
        check(lc.draft_carry(pending) == nullptr,     "cold: draft_carry() refuses the row");
        check(lc.target_process_mode(0)  == pmode::cold_zero,
              "cold: frontier 0 is cold_zero");
        check(lc.target_process_mode(7)  == pmode::target_only,
              "cold: nonzero frontier is target_only");
        check(lc.target_process_mode(-1) == pmode::target_only,
              "cold: negative frontier is target_only");
    }

    // ------------------------------------------------------------- warm
    printf("== refreshing the target makes the carry consumable ==\n");
    {
        lifecycle lc;
        const float pending[2] = { 1.0f, 2.0f };
        lc.target_process_refreshed();
        check(lc.draft_ready(),                        "warm: draft_ready() is true");
        check(lc.draft_carry(pending) == pending,      "warm: draft_carry() hands back the row");
        check(lc.target_process_mode(0) == pmode::retained_carry,
              "warm: frontier 0 is retained_carry");
        check(lc.target_process_mode(7) == pmode::retained_carry,
              "warm: nonzero frontier stays retained_carry (retained beats cold)");

        lc.target_process_skipped();
        check(!lc.draft_ready(),                       "skipped: draft_ready() is false again");
        check(lc.draft_carry(pending) == nullptr,      "skipped: draft_carry() refuses again");
        check(lc.target_process_mode(0) == pmode::cold_zero,
              "skipped: frontier 0 is cold_zero again");
    }

    // -------------------------------------------------- every transition
    printf("== every sequence transition discards the carry ==\n");
    for (size_t i = 0; i < sizeof(k_events)/sizeof(k_events[0]); i++) {
        lifecycle lc;
        lc.target_process_refreshed();
        lc.sequence_transition(k_events[i]);
        char buf[80];
        snprintf(buf, sizeof(buf), "event %u drops the carry", (unsigned) i);
        check(!lc.draft_ready(), buf);
    }

    // ------------------------------------------------------------- codec
    printf("== the codec round trip ==\n");
    {
        const std::vector<float> source = { 0.5f, -1.25f, 3.0f };
        std::vector<uint8_t> data;

        lifecycle writer;
        writer.target_process_refreshed();
        check(common_speculative_mtp_carry_state_save(writer, source, data),
              "save: a ready carry with a payload is accepted");

        // frozen golden wire format: magic, version, width, then the floats
        const size_t header_size = 3*sizeof(uint32_t);
        check(data.size() == header_size + source.size()*sizeof(float),
              "wire: total size is header + 4 bytes per float");
        uint32_t magic = 0, version = 0, width = 0;
        const uint32_t want_magic = 0x4d545043; // MTPC
        std::memcpy(&magic,   data.data() + 0*sizeof(uint32_t), sizeof(uint32_t));
        std::memcpy(&version, data.data() + 1*sizeof(uint32_t), sizeof(uint32_t));
        std::memcpy(&width,   data.data() + 2*sizeof(uint32_t), sizeof(uint32_t));
        check(magic   == want_magic,              "wire: magic is 0x4d545043 (MTPC)");
        check(version == 1,                       "wire: version is 1");
        check(width   == (uint32_t) source.size(),"wire: width is the element count");
        std::vector<float> frozen(source.size());
        std::memcpy(frozen.data(), data.data() + header_size, source.size()*sizeof(float));
        check(frozen == source,                   "wire: the payload is the floats verbatim");

        // the codec checks the declared width against the DESTINATION size, so
        // the caller must size the row before loading into it.
        std::vector<float> restored(source.size());
        lifecycle reader;
        check(common_speculative_mtp_carry_state_load(reader, restored, data),
              "load: a well-formed image is accepted");
        check(restored == source,                 "load: the payload comes back unchanged");
        // load() clears first and only refreshes after a full validation, so a
        // half-applied image can never leave a consumable carry behind.
        check(reader.draft_ready(),
              "load: a fully validated image leaves the carry ready");
    }

    // ------------------------------------------------------- refusals
    printf("== the codec refuses every malformed image ==\n");
    {
        const std::vector<float> source = { 1.0f, 2.0f };
        std::vector<uint8_t> good;
        lifecycle writer;
        writer.target_process_refreshed();
        check(common_speculative_mtp_carry_state_save(writer, source, good),
              "refusals: the good image is built");

        // save: not ready
        {
            std::vector<uint8_t> data(7, 0xAB);
            lifecycle cold;
            check(!common_speculative_mtp_carry_state_save(cold, source, data),
                  "save: refuses when the carry is not ready");
            check(data.empty(), "save: a refusal clears the output buffer");
        }
        // save: empty payload
        {
            std::vector<uint8_t> data;
            lifecycle warm;
            warm.target_process_refreshed();
            check(!common_speculative_mtp_carry_state_save(warm, {}, data),
                  "save: refuses an empty payload");
            check(data.empty(), "save: an empty payload clears the output buffer");
        }
        // load: truncated
        {
            std::vector<uint8_t> data(good.begin(), good.begin() + 8);
            std::vector<float> out;
            lifecycle lc;
            check(!common_speculative_mtp_carry_state_load(lc, out, data),
                  "load: refuses a truncated header");
            check(!lc.draft_ready(), "load: a truncated image leaves the carry cold");
        }
        // load: wrong magic
        {
            std::vector<uint8_t> data = good;
            const uint32_t bad = 0xDEADBEEF;
            std::memcpy(data.data(), &bad, sizeof(uint32_t));
            std::vector<float> out;
            lifecycle lc;
            check(!common_speculative_mtp_carry_state_load(lc, out, data),
                  "load: refuses a foreign magic");
            check(!lc.draft_ready(), "load: a foreign magic leaves the carry cold");
        }
        // load: wrong version
        {
            std::vector<uint8_t> data = good;
            const uint32_t bad = 2;
            std::memcpy(data.data() + sizeof(uint32_t), &bad, sizeof(uint32_t));
            std::vector<float> out;
            lifecycle lc;
            check(!common_speculative_mtp_carry_state_load(lc, out, data),
                  "load: refuses a future version");
            check(!lc.draft_ready(), "load: a future version leaves the carry cold");
        }
        // load: width disagreeing with the destination
        {
            std::vector<uint8_t> data = good;
            const uint32_t bad = 99;
            std::memcpy(data.data() + 2*sizeof(uint32_t), &bad, sizeof(uint32_t));
            std::vector<float> out(2, 0.0f);   // destination says 2, image says 99
            lifecycle lc;
            check(!common_speculative_mtp_carry_state_load(lc, out, data),
                  "load: refuses a width that disagrees with the destination");
            check(!lc.draft_ready(), "load: a width mismatch leaves the carry cold");
        }
        // load: trailing bytes (width and destination agree, so only the total
        // length can reject this one)
        {
            std::vector<uint8_t> data = good;
            data.push_back(0);
            std::vector<float> out(2, 0.0f);
            lifecycle lc;
            check(!common_speculative_mtp_carry_state_load(lc, out, data),
                  "load: refuses trailing bytes");
            check(!lc.draft_ready(), "load: trailing bytes leave the carry cold");
        }
        // load: a refusal must not publish a half row
        {
            std::vector<float> out(2, 7.0f);
            std::vector<uint8_t> data(good.begin(), good.begin() + 14); // header + a partial float
            lifecycle lc;
            check(!common_speculative_mtp_carry_state_load(lc, out, data),
                  "load: the 14-byte image is refused");
            check(out[0] == 7.0f && out[1] == 7.0f,
                  "load: a refused image leaves the destination untouched");
        }
    }

    // ------------------------------------------------- negative control
    // Proves the counter and the exit code can actually move. The injected
    // failure is rolled back so it never reaches the reported total.
    printf("== negative control: the counter really can fail ==\n");
    {
        const int checks_before = g_checks;
        const int fails_before  = g_fails;
        check(false, "*** expected FAIL: this check is deliberately false ***");
        const bool counted = (g_fails == fails_before + 1);
        g_checks = checks_before;
        g_fails  = fails_before;
        check(counted, "the failure counter moved by exactly one");
    }

    printf("\n%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
