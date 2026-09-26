#include "llama-vbr-explicit-capture.h"
#include "llama-sha256.h"
#include "turbo-rotation-data.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct test_env {
    const char * name;
    bool present;
    std::string saved;
    explicit test_env(const char * n) : name(n), present(std::getenv(n) != nullptr),
        saved(present ? std::getenv(n) : "") { set(nullptr); }
    void set(const char * value) const {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
    }
    ~test_env() { set(present ? saved.c_str() : nullptr); }
};

// Independent, uncached serialization of the existing rotation wire identity.
static std::array<uint8_t, 32> rotation(int type, bool value_side) {
    llama_sha256_writer writer;
    constexpr char tag[] = "buun.vbr.codec-rotation/v1";
    writer.string(tag, sizeof(tag)-1);
    writer.u32(type);
    writer.u32(value_side);
    const float * matrix = value_side ? TURBO_ROTATION_RT : TURBO_ROTATION_R;
    writer.bytes(matrix, 128*128*sizeof(float));
    return writer.finish();
}

int main() {
    try {
        // No environment writes while worker threads run.
        test_env t8("TURBO_CB_T8"), t4("TURBO_CB_T4"), tcq("TURBO_TCQ_CB"),
            tcq_k("TURBO_TCQ_CB_K"), tcq_v("TURBO_TCQ_CB_V"), t1("TURBO1_TCQ_CB"),
            t1_k("TURBO1_TCQ_CB_K"), t1_v("TURBO1_TCQ_CB_V"),
            mean_off("TURBO_MEANSUB_OFF"), mean_k("TURBO_KMEAN_SUB"), mean_v("TURBO_VMEAN_SUB");
        const vbr_explicit_representation_policy policy {"identity-test", 13};
        const auto identity = [&](int type, bool side) {
            vbr_explicit_representation_identity result;
            check(vbr_explicit_capture_representation_identity(&policy, type, side, 0, result), "identity failed");
            return result;
        };

        // Concurrent FIRST use of each side, not just concurrent warm lookups.
        const std::array<std::array<uint8_t, 32>, 2> expected {
            rotation(GGML_TYPE_TURBO4_0, false), rotation(GGML_TYPE_TURBO4_0, true)};
        std::atomic<int> ready {0};
        std::atomic<bool> go {false}, ok {true};
        std::vector<std::thread> workers;
        for (int i = 0; i < 8; ++i) {
            workers.emplace_back([&, i] {
                ++ready;
                while (!go.load()) { std::this_thread::yield(); }
                for (int repeat = 0; repeat < 16; ++repeat) {
                    vbr_explicit_representation_identity result;
                    if (!vbr_explicit_capture_representation_identity(&policy, GGML_TYPE_TURBO4_0, i&1, 0, result) ||
                        result.rotation_digest != expected[i&1]) { ok = false; }
                }
            });
        }
        while (ready.load() != 8) { std::this_thread::yield(); }
        go = true;
        for (auto & worker : workers) { worker.join(); }
        check(ok, "concurrent first-use identity differs");

        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            for (bool side : {false, true}) {
                const auto reference = rotation(type, side);
                check(identity(type, side).rotation_digest == reference, "rotation wire identity changed");
                check(identity(type, side).rotation_digest == reference, "warm rotation identity changed");
            }
        }
        const auto original = identity(GGML_TYPE_TURBO4_0, false);
        const vbr_explicit_representation_policy other_build {"other-build", 11};
        vbr_explicit_representation_identity changed;
        check(vbr_explicit_capture_representation_identity(&other_build, GGML_TYPE_TURBO4_0, false, 0, changed), "other build failed");
        check(changed.codebook_digest != original.codebook_digest && changed.rotation_digest == original.rotation_digest,
              "build identity was cached with rotation");
        mean_off.set("1");
        changed = identity(GGML_TYPE_TURBO4_0, false);
        check(changed.meansub_digest != original.meansub_digest && changed.rotation_digest == original.rotation_digest,
              "mean setting was cached with rotation");
        mean_off.set(nullptr);

#ifdef __linux__
        // Anonymous temporary backing lets the same override path change bytes
        // without touching any user file. Both override readers must stay fresh.
        const auto close_file = [](FILE * file) { fclose(file); };
        std::unique_ptr<FILE, decltype(close_file)> file(tmpfile(), close_file);
        check(bool(file), "temporary override creation failed");
        const std::string path = "/proc/self/fd/" + std::to_string(fileno(file.get()));
        const auto write = [&](const char * bytes) {
            check(fseek(file.get(), 0, SEEK_SET) == 0 && fwrite(bytes, 1, 4, file.get()) == 4 &&
                  fflush(file.get()) == 0, "temporary override write failed");
        };
        t4.set(path.c_str()); mean_k.set(path.c_str());
        write("AAAA");
        const auto first = identity(GGML_TYPE_TURBO4_0, false);
        write("BBBB");
        const auto second = identity(GGML_TYPE_TURBO4_0, false);
        check(first.codebook_digest != second.codebook_digest && first.meansub_digest != second.meansub_digest &&
              first.rotation_digest == second.rotation_digest, "same-path override edit was missed");
        file.reset();
        check(!vbr_explicit_capture_representation_identity(&policy, GGML_TYPE_TURBO4_0, false, 0, changed),
              "missing override accepted after warm lookup");
#endif
        check(!vbr_explicit_capture_representation_identity(&policy, -1, false, 0, changed), "negative type accepted");
        check(!vbr_explicit_capture_representation_identity(&policy, GGML_TYPE_COUNT, false, 0, changed), "invalid type accepted");
        std::puts("representation identity PASS: exact wire bytes, cold concurrency, fresh mutable settings");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "representation identity FAIL: %s\n", e.what());
        return 1;
    }
}
