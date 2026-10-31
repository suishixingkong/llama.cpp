// Layer 2 of the conversation store: the adapter's llama_kvmem_store_* handles.
//
// Every one of those fifteen functions was ported with the adapter, but nothing
// ever exercised them here, so "the mechanism exists" was the only claim behind
// layer 1's policy. This test is the missing claim: on a real model it creates a
// second host store, decodes into each one in turn, and checks that switching
// back returns the rows it left behind.
//
// What it can and cannot prove:
//   - it proves the pool, the drain and the restage work on CUDA with flash
//     attention, which is the configuration conv_swap_supported() requires;
//   - it does NOT prove the recurrent half (a hybrid model's byte snapshot) is
//     carried across a swap: that state belongs to the caller, and swapping only
//     rebinds KV. A text-only transformer has none, which is what this test runs.
//
// Needs a model, so it prints SKIP and exits 0 without one. That keeps ctest
// green on a machine that has no GGUF, the same way the other model-backed
// tests here behave.
//
//   ./bin/kvmem-store-swap-test /path/to/model.gguf [device_index]

#include "llama.h"
#include "llama-kvmem-hooks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char * what, long long got, long long want) {
    std::printf("%-4s %-44s got %lld want %lld\n", ok ? "ok" : "FAIL", what, got, want);
    if (!ok) {
        ++g_failures;
    }
}

// For "any non-empty store", where a fixed expected count would only restate
// how the tokenizer split the sample text.
static void check_positive(bool ok, const char * what, long long got) {
    std::printf("%-4s %-44s got %lld (want > 0)\n", ok ? "ok" : "FAIL", what, got);
    if (!ok) {
        ++g_failures;
    }
}

static void fail(const char * what) {
    std::printf("FAIL %s\n", what);
    ++g_failures;
}

// Decode `n` tokens taken from `text` and leave the sequence at `n_past`.
// llama_batch_get_one tracks the position of sequence 0, so each call appends.
static bool decode_text(llama_context * ctx, const llama_vocab * vocab, const char * text, int32_t n) {
    std::vector<llama_token> tokens((size_t) n);
    int32_t count = llama_tokenize(vocab, text, (int32_t) std::strlen(text), tokens.data(), n, true, false);
    if (count < 0) {
        // Buffer was short: the negative result is the required size.
        tokens.resize((size_t) -count);
        count = llama_tokenize(vocab, text, (int32_t) std::strlen(text), tokens.data(), (int32_t) tokens.size(), true, false);
    }
    if (count <= 0) {
        return false;
    }
    const int32_t take = count < n ? count : n;
    llama_batch batch = llama_batch_get_one(tokens.data(), take);
    return llama_decode(ctx, batch) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::printf("SKIP: usage: %s <model.gguf>\n", argv[0]);
        return 0;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    // Pin one device: a layered split would put half the blocks on the second
    // GPU, and this test is about the host store, not about multi-GPU split.
    // A build whose CUDA arch does not cover the small card must still run, so
    // the device is given rather than discovered.
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    mparams.main_gpu   = argc > 2 ? std::atoi(argv[2]) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        std::printf("FAIL cannot load %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    // Arm KVMem before the context exists: the setter is process-wide and its
    // contract is "call before llama_init_from_model". The budget is left at 0
    // so the whole context stays resident and eviction cannot move rows behind
    // the assertions.
    llama_kvmem_params kp = {};
    kp.enabled            = true;
    kp.block_tokens       = 32;
    kp.budget             = 0;
    kp.method             = 0; // recency: the cheapest selector, no scoring pass
    kp.payload_host_bytes = (uint64_t) 4 << 30;
    llama_kvmem_set_params(&kp);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = 4096;
    cparams.n_batch        = 512;
    cparams.n_ubatch       = 512;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::printf("FAIL cannot create the context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const bool supported = llama_kvmem_store_swap_supported();
    check(supported, "store swap supported", supported ? 1 : 0, 1);
    if (!supported) {
        std::printf("FAIL a CUDA flash-attention build must support host-store swap\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // The legacy handle before any create(): the store the memory object built.
    check(llama_kvmem_store_current() == 0, "current store before create", llama_kvmem_store_current(), 0);

    // First create() binds the pool and installs entry 0, which is the active
    // store's placeholder; handles start at 1.
    const int32_t a = llama_kvmem_store_create();
    check(a >= 1, "create store A", a, 1);
    if (a < 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Switching to A retires the constructor's store, which belongs to no
    // conversation: the pool keeps it as entry 0 and A comes in empty.
    const bool attached_a = llama_kvmem_store_switch(a);
    check(!attached_a, "switch to an empty store A reports no rows", attached_a ? 1 : 0, 0);
    check(llama_kvmem_store_current() == a, "current store after switching to A", llama_kvmem_store_current(), a);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (!decode_text(ctx, vocab, "The quick brown fox jumps over the lazy dog. ", 24)) {
        fail("decode into store A");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const uint32_t rows_a = llama_kvmem_store_rows(a);
    check_positive(rows_a > 0, "store A holds rows after decoding", rows_a);
    const uint64_t bytes_a = llama_kvmem_store_bytes(a);
    check_positive(bytes_a > 0, "store A accounts host bytes", (long long) bytes_a);

    // B arrives empty; A's rows must survive living in the pool while B runs.
    const int32_t b = llama_kvmem_store_create();
    check(b > a, "create store B", b, a + 1);
    const bool attached_b = llama_kvmem_store_switch(b);
    check(!attached_b, "switch to a fresh store B reports no rows", attached_b ? 1 : 0, 0);
    check(llama_kvmem_store_current() == b, "current store after switching to B", llama_kvmem_store_current(), b);
    check(llama_kvmem_store_rows(b) == 0, "store B starts empty", llama_kvmem_store_rows(b), 0);

    // A is parked, not gone: this is the property layer 3 depends on.
    check(llama_kvmem_store_rows(a) == rows_a, "store A still holds its rows while parked",
          llama_kvmem_store_rows(a), rows_a);

    if (!decode_text(ctx, vocab, "A second conversation runs on the same GPU. ", 16)) {
        fail("decode into store B");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const uint32_t rows_b = llama_kvmem_store_rows(b);
    check_positive(rows_b > 0, "store B holds rows after decoding", rows_b);
    check(llama_kvmem_store_rows(a) == rows_a, "store A is untouched by B's decode",
          llama_kvmem_store_rows(a), rows_a);

    // Switch back: the one claim that matters is that A is restaged with its
    // rows, not that the call returns anything in particular.
    const bool restaged = llama_kvmem_store_switch(a);
    check(restaged, "switching back to A restages it", restaged ? 1 : 0, 1);
    check(llama_kvmem_store_current() == a, "current store after switching back", llama_kvmem_store_current(), a);
    check(llama_kvmem_store_rows(a) == rows_a, "store A holds its rows after the restage",
          llama_kvmem_store_rows(a), rows_a);
    check(llama_kvmem_store_rows(b) == rows_b, "store B keeps its rows while parked",
          llama_kvmem_store_rows(b), rows_b);

    // Continuing on the restaged store must decode, which is what a server's
    // next turn does after a swap: the rows are only useful if the context can
    // append to them.
    if (!decode_text(ctx, vocab, "and it continues from the restored prefix. ", 8)) {
        fail("decode on the restaged store A");
    } else {
        check(llama_kvmem_store_rows(a) == rows_a + 8, "store A appends after the restage",
              llama_kvmem_store_rows(a), rows_a + 8);
    }

    // The active store cannot be destroyed: there would be no store to run on.
    // Each handle is asked once and the answer asserted from the variable: a
    // second call inside the same argument list would race its own side effect.
    const bool refused_active = llama_kvmem_store_destroy(a);
    check(!refused_active, "destroying the active store is refused", refused_active ? 1 : 0, 0);
    const bool killed_parked = llama_kvmem_store_destroy(b);
    check(killed_parked, "destroying a parked store succeeds", killed_parked ? 1 : 0, 1);
    check(llama_kvmem_store_rows(b) == 0, "a destroyed store reports no rows", llama_kvmem_store_rows(b), 0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::printf("\n%s\n", g_failures == 0 ? "store create/switch/rows PASS" : "store create/switch/rows FAILED");
    return g_failures == 0 ? 0 : 1;
}
