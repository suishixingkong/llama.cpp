// Runtime check for the LoRA execution-content digest (slice 7d precondition).
//
// llama_adapter_meta_digest() keys persisted prompt-cache state: two adapters that
// execute identically must hash identically across process restarts, and any change
// that alters execution (tensor bytes, tensor shapes, semantic metadata, alpha) must
// change the hash. A digest that is wrong in the *permissive* direction (two distinct
// adapter sets colliding) silently reuses a cache entry built for a different adapter
// -- exactly the failure mode the port must not ship. So this test is built around
// negative controls, not around a golden captured from one particular model.
//
// The adapters are synthesized at run time from the real base model (the upstream
// lora test suite does the same); the expected digest is recomputed from the written
// .gguf by an independent implementation in the local harness script
// verify_lora_digest.py, which is what catches field-order and serialization
// mistakes. This file prints the digest in hex for that cross-check.
//
// Deliberately NOT assert-based: the Release build defines NDEBUG, which turns
// assert() into a no-op and an assert-shaped test into a vacuous one. Failures are
// counted and printed, and the exit code is derived from that count.
//
// usage: test-lora-adapter-digest MODEL ADAPTER_DIR

#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

struct lora_target {
    const char * name;
    uint32_t     seed;
};

// Candidate LoRA target tensors, most representative first. output.weight is the
// tensor the upstream fixture uses, but a model with tied word embeddings does not
// expose it, so fall back to projections that always exist.
static const char * pick_target_tensor(const llama_model * model, size_t skip_this_many = 0) {
    static const char * candidates[] = {
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_gate.weight",
        "token_embd.weight",
    };
    for (const char * name : candidates) {
        const ggml_tensor * t = model->get_tensor(name);
        if (t != nullptr && ggml_n_dims(t) == 2) {
            if (skip_this_many > 0) {
                --skip_this_many;
                continue;
            }
            return name;
        }
    }
    return nullptr;
}

// Writes a minimal LoRA gguf holding one (a, b) pair per target, in the order given.
// Each pair's bytes depend only on its own seed, so the same set of targets written
// in a different order is the same adapter content -- which is what lets the caller
// check that the digest canonicalizes the order.
static void write_adapter(
        const char * path,
        const llama_model * model,
        const std::vector<lora_target> & targets,
        float alpha) {
    const int64_t rank = 4;

    size_t data_size = 0;
    for (const lora_target & target : targets) {
        const ggml_tensor * base = model->get_tensor(target.name);
        GGML_ASSERT(base != nullptr);
        GGML_ASSERT(ggml_n_dims(base) == 2);
        data_size += size_t(base->ne[0] + base->ne[1])*rank*sizeof(float);
    }

    gguf_context_ptr gguf(gguf_init_empty());
    gguf_set_val_str(gguf.get(), "general.type", "adapter");
    gguf_set_val_str(gguf.get(), "general.architecture", llm_arch_name(model->arch));
    gguf_set_val_str(gguf.get(), "adapter.type", "lora");
    gguf_set_val_f32(gguf.get(), "adapter.lora.alpha", alpha);

    ggml_init_params params = {
        /*.mem_size   =*/ data_size + 2*targets.size()*ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx != nullptr);

    for (const lora_target & target : targets) {
        const ggml_tensor * base = model->get_tensor(target.name);

        ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, base->ne[0], rank);
        ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rank, base->ne[1]);
        ggml_set_name(a, (std::string(target.name) + ".lora_a").c_str());
        ggml_set_name(b, (std::string(target.name) + ".lora_b").c_str());

        uint32_t seed = target.seed;
        auto fill = [&](ggml_tensor * tensor) {
            float * data = static_cast<float *>(tensor->data);
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                seed = 1664525u*seed + 1013904223u;
                data[i] = (float(int32_t(seed >> 8) % 2001) - 1000.0f)*1.0e-5f;
            }
        };
        fill(a);
        fill(b);

        gguf_add_tensor(gguf.get(), a);
        gguf_add_tensor(gguf.get(), b);
    }

    gguf_write_to_file(gguf.get(), path, false);
}

static std::array<uint8_t, 32> digest_of(llama_adapter_lora * adapter) {
    std::array<uint8_t, 32> digest{};
    llama_adapter_meta_digest(adapter, digest.data());
    return digest;
}

static bool all_zero(const std::array<uint8_t, 32> & digest) {
    for (uint8_t byte : digest) {
        if (byte != 0) {
            return false;
        }
    }
    return true;
}

static std::string to_hex(const std::array<uint8_t, 32> & digest) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t byte : digest) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

// Loads ADAPTER and returns its digest, or a zero digest when the load failed (in
// which case the caller's CHECK already recorded the failure).
static std::array<uint8_t, 32> digest_of_file(
        llama_model * model,
        const std::string & path,
        bool * loaded) {
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());
    *loaded = adapter != nullptr;
    if (adapter == nullptr) {
        return {};
    }
    const std::array<uint8_t, 32> digest = digest_of(adapter);
    llama_adapter_lora_free(adapter);
    return digest;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL ADAPTER_DIR\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    const std::string adapter_dir = argv[2];

    std::string target;
    std::string second_target;
    std::string hex_base;
    std::string hex_other_data;
    std::string hex_other_alpha;
    std::string hex_other_tensor;
    std::string hex_two_fwd;
    std::string hex_two_rev;

    std::array<uint8_t, 32> digest_base{};

    {
        std::printf("=== pass 1: load the base model ===\n");
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model_ptr model(llama_model_load_from_file(argv[1], model_params));
        CHECK(model != nullptr);
        if (model) {
            const char * target_name = pick_target_tensor(model.get());
            const char * second_name = pick_target_tensor(model.get(), 1);
            CHECK(target_name != nullptr);
            if (target_name != nullptr) {
                target = target_name;
                std::printf("target tensor: %s\n", target.c_str());
                if (second_name != nullptr) {
                    second_target = second_name;
                    std::printf("second tensor: %s\n", second_target.c_str());
                }

                auto path_of = [&](const char * name) {
                    return adapter_dir + "/" + name;
                };

                write_adapter(path_of("digest-base.gguf").c_str(), model.get(),
                        { { target_name, 0x12345678u } }, 4.0f);
                // identical metadata and shapes, different tensor bytes
                write_adapter(path_of("digest-other-data.gguf").c_str(), model.get(),
                        { { target_name, 0x0badc0deu } }, 4.0f);
                // identical tensor bytes (same seed), different alpha
                write_adapter(path_of("digest-other-alpha.gguf").c_str(), model.get(),
                        { { target_name, 0x12345678u } }, 8.0f);
                if (second_name != nullptr) {
                    // a different tensor name, i.e. a different shape
                    write_adapter(path_of("digest-other-tensor.gguf").c_str(), model.get(),
                            { { second_name, 0x12345678u } }, 4.0f);
                    // the same two pairs, recorded in the opposite order in the file:
                    // the loader stores them in an unordered_map, so only the digest's
                    // own canonical ordering can make these two agree
                    write_adapter(path_of("digest-two-fwd.gguf").c_str(), model.get(),
                            { { target_name, 0x11111111u }, { second_name, 0x22222222u } }, 4.0f);
                    write_adapter(path_of("digest-two-rev.gguf").c_str(), model.get(),
                            { { second_name, 0x22222222u }, { target_name, 0x11111111u } }, 4.0f);
                }

                bool ok = false;
                digest_base        = digest_of_file(model.get(), path_of("digest-base.gguf"), &ok);
                CHECK(ok);
                const auto digest_other_data  = digest_of_file(model.get(), path_of("digest-other-data.gguf"), &ok);
                CHECK(ok);
                const auto digest_other_alpha = digest_of_file(model.get(), path_of("digest-other-alpha.gguf"), &ok);
                CHECK(ok);

                // the digest must be a real hash, not a zero-initialized field
                CHECK(!all_zero(digest_base));
                // negative controls: tensor bytes and alpha are both load-bearing
                CHECK(digest_base != digest_other_data);
                CHECK(digest_base != digest_other_alpha);
                CHECK(digest_other_data != digest_other_alpha);

                hex_base         = to_hex(digest_base);
                hex_other_data   = to_hex(digest_other_data);
                hex_other_alpha  = to_hex(digest_other_alpha);
                std::printf("digest base       = %s\n", hex_base.c_str());
                std::printf("digest other-data = %s\n", hex_other_data.c_str());
                std::printf("digest other-alpha= %s\n", hex_other_alpha.c_str());

                if (second_name != nullptr) {
                    const auto digest_other_tensor = digest_of_file(model.get(), path_of("digest-other-tensor.gguf"), &ok);
                    CHECK(ok);
                    CHECK(digest_base != digest_other_tensor);
                    hex_other_tensor = to_hex(digest_other_tensor);
                    std::printf("digest other-tens = %s  (%s)\n", hex_other_tensor.c_str(), second_target.c_str());

                    const auto digest_two_fwd = digest_of_file(model.get(), path_of("digest-two-fwd.gguf"), &ok);
                    CHECK(ok);
                    const bool fwd_ok = ok;
                    const auto digest_two_rev = digest_of_file(model.get(), path_of("digest-two-rev.gguf"), &ok);
                    CHECK(ok);
                    if (fwd_ok && ok) {
                        CHECK(digest_two_fwd == digest_two_rev);
                        CHECK(digest_two_fwd != digest_base);
                        CHECK(digest_two_fwd != digest_other_tensor);
                    }
                    hex_two_fwd = to_hex(digest_two_fwd);
                    hex_two_rev = to_hex(digest_two_rev);
                    std::printf("digest two-fwd    = %s\n", hex_two_fwd.c_str());
                    std::printf("digest two-rev    = %s\n", hex_two_rev.c_str());
                }

                // loading the same file twice must produce the same digest: the
                // loader keys weights by an unordered_map, so this pins that the
                // digest canonicalizes rather than depending on container order
                const auto digest_base_again = digest_of_file(model.get(), path_of("digest-base.gguf"), &ok);
                CHECK(ok);
                CHECK(digest_base_again == digest_base);
            }
        }
    }

    // The digest is the key of a *persisted* cache, so it has to survive a process
    // restart: reload the model from scratch and the same file must hash the same.
    if (!hex_base.empty()) {
        std::printf("=== pass 2: reload the model in a second context ===\n");
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model_ptr model(llama_model_load_from_file(argv[1], model_params));
        CHECK(model != nullptr);
        if (model) {
            bool ok = false;
            const auto digest_reloaded = digest_of_file(model.get(), adapter_dir + "/digest-base.gguf", &ok);
            CHECK(ok);
            CHECK(digest_reloaded == digest_base);
            if (digest_reloaded != digest_base) {
                std::printf("digest reloaded   = %s\n", to_hex(digest_reloaded).c_str());
            }
        }
    }

    llama_backend_free();

    // machine-readable lines for the Python cross-check
    std::printf("RESULT target=%s\n", target.c_str());
    std::printf("RESULT second_target=%s\n", second_target.c_str());
    std::printf("RESULT digest_base=%s\n", hex_base.c_str());
    std::printf("RESULT digest_other_data=%s\n", hex_other_data.c_str());
    std::printf("RESULT digest_other_alpha=%s\n", hex_other_alpha.c_str());
    std::printf("RESULT digest_other_tensor=%s\n", hex_other_tensor.c_str());
    std::printf("RESULT digest_two_fwd=%s\n", hex_two_fwd.c_str());
    std::printf("RESULT digest_two_rev=%s\n", hex_two_rev.c_str());
    std::printf("test-lora-adapter-digest: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
