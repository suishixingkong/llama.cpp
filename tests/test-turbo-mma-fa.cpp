// Differential test for the opt-in TurboQuant MMA FlashAttention path (Stage 1c L1,
// BUUN_MERGE_PLAN.md sections 1.4a / 10.9).
//
// What it verifies, per (turbo KV type, shape):
//   1. Dispatch: with GGML_CUDA_TURBO_MMA=1 every case below must increment the
//      ggml_backend_cuda_get_turbo_mma_fa_invocations() counter (i.e. the op really ran on
//      the turbo MMA kernel, not silently on the regular VEC/TILE/MMA_F16 path). Without the
//      env var ("fallback" mode) the counter must stay at 0.
//   2. Numerics: the MMA kernel decodes the same TurboQuant KV blocks the CPU reference
//      decodes, so the two outputs may only differ by in-kernel fp16 tile rounding and
//      accumulation order -- asserted to stay far below quantization-error scale, and far
//      below what a wrong tile/group addressing bug (O(1) garbage) would produce.
//
// The shapes cover every tile the Volta dispatch ladder can select for D in {128, 256}:
// gqa 8/4/2/1 -> ncols2 8/4/2/1, small and large nq -> 32-column and 64-column rungs, and a
// no-mask case -> the ncols2=1 non-GQA path. No model file needed.
//
// Usage: test-turbo-mma-fa [fallback]
//   (default)  sets GGML_CUDA_TURBO_MMA=1 and asserts the MMA counter advances.
//   fallback   leaves the env var off and asserts the counter never advances; the numeric
//              comparison still runs against whatever regular kernel the backend picks.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "testing.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t NKV = 512; // multiple of FATTN_KQ_STRIDE (256) so the GQA-optimized path applies

struct fa_shape {
    int64_t    D;
    int64_t    nq;
    int64_t    nhq;
    int64_t    nhkv;
    bool       mask;
    const char * label;
};

// Every rung of the Volta turbo dispatch ladder (see fattn-mma-turbo.cuh):
const fa_shape shapes[] = {
    // gqa=8 -> ncols2=8: 32-column rung (nq <= 32/8) and 64-column rung
    {128,  4, 8, 1, true,  "D=128 gqa=8 nq=4  tile(4,8)"},
    {128, 32, 8, 1, true,  "D=128 gqa=8 nq=32 tile(8,8) 64-rung"},
    // gqa=4 -> ncols2=4, gqa=2 -> ncols2=2, gqa=1 -> ncols2=1
    {128,  8, 8, 2, true,  "D=128 gqa=4 nq=8  tile(8,4)"},
    {128, 16, 4, 2, true,  "D=128 gqa=2 nq=16 tile(16,2)"},
    {128, 16, 2, 2, true,  "D=128 gqa=1 nq=16 tile(32,1)"},
    // no mask -> use_gqa_opt=false -> ncols2=1 path
    {128,  4, 8, 1, false, "D=128 gqa=8 nq=4  no-mask tile(32,1)"},
    // D=256 (two 128-element rotation groups per row)
    {256,  4, 8, 1, true,  "D=256 gqa=8 nq=4  tile(4,8)"},
    {256, 32, 8, 1, true,  "D=256 gqa=8 nq=32 tile(8,8) 64-rung"},
};

std::vector<float> gen_f32(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t rng = seed;
    for (size_t i = 0; i < n; ++i) {
        rng = rng*1664525u + 1013904223u;
        const float u = float(rng >> 8)*(1.0f/16777216.0f)*2.0f - 1.0f;
        // Wide dynamic range plus a slow sine, so the trellis/codec sees more than a smooth ramp.
        v[i] = u*3.0f + 0.25f*std::sin(float(i)*0.05f);
    }
    return v;
}

// Causal mask in f16 bytes: row i attends KV columns <= i (+ optional sink-free bias 0).
std::vector<uint8_t> gen_mask_f16(int64_t nq, int64_t nkv) {
    std::vector<uint8_t> m(nq*nkv*sizeof(uint16_t));
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < nkv; ++j) {
            const float v = j <= i ? 0.0f : -INFINITY;
            const uint16_t h = ggml_fp32_to_fp16(v);
            memcpy(m.data() + (i*nkv + j)*sizeof(uint16_t), &h, sizeof(h));
        }
    }
    return m;
}

// Builds and runs one FLASH_ATTN_EXT with turbo-quantized K/V on `backend`, returns the f32 output.
std::vector<float> run_fa(
        ggml_backend_t backend,
        ggml_type kv_type,
        const fa_shape & s,
        const std::vector<float> & q_src,
        const std::vector<uint8_t> & k_bytes,
        const std::vector<uint8_t> & v_bytes,
        const std::vector<uint8_t> * mask_bytes) {
    constexpr size_t N_TENSORS = 16;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false);
    const ggml_init_params params{context_bytes, nullptr, true};

    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,  s.D, s.nq, s.nhq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), kv_type,        s.D, NKV, s.nhkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), kv_type,        s.D, NKV, s.nhkv, 1);
    ggml_tensor * mask = mask_bytes ?
        ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, NKV, s.nq, 1, 1) : nullptr;

    // No sinks operand; logit_softcap = 0 disables it.
    const float scale = 1.0f/std::sqrt((float) s.D);
    ggml_tensor * kqvb = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, scale, 0.0f, 0.0f);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);

    ggml_backend_tensor_set(q, q_src.data(), 0, q_src.size()*sizeof(float));
    ggml_backend_tensor_set(k, k_bytes.data(), 0, k_bytes.size());
    ggml_backend_tensor_set(v, v_bytes.data(), 0, v_bytes.size());
    if (mask_bytes) {
        ggml_backend_tensor_set(mask, mask_bytes->data(), 0, mask_bytes->size());
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, kqvb);
    GGML_ASSERT(ggml_backend_supports_op(backend, kqvb));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> out(ggml_nelements(kqvb));
    GGML_ASSERT(ggml_is_contiguous(kqvb));
    ggml_backend_tensor_get(kqvb, out.data(), 0, out.size()*sizeof(float));
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    testing t;

    const bool mma_mode = !(argc > 1 && std::string(argv[1]) == "fallback");
    if (mma_mode) {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_TURBO_MMA", "1");
#else
        setenv("GGML_CUDA_TURBO_MMA", "1", 1);
#endif
    }
    printf("mode: %s (GGML_CUDA_TURBO_MMA=%s)\n", mma_mode ? "mma" : "fallback",
           mma_mode ? "1" : "unset");

    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    if (!cuda_backend) {
        printf("no CUDA device available - skipping\n");
        return 0;
    }
    ggml_backend_ptr cpu_backend(ggml_backend_cpu_init());
    GGML_ASSERT(cpu_backend);

    const ggml_type types[] = {
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ,
    };

    for (ggml_type kv_type : types) {
        for (const fa_shape & s : shapes) {
            std::string name = std::string(ggml_type_name(kv_type)) + " " + s.label;
            t.test(name.c_str(), [&](testing & tt) {
                const size_t kv_row_bytes = ggml_row_size(kv_type, s.D);
                const size_t kv_bytes     = kv_row_bytes*(size_t)(NKV*s.nhkv);

                // Identical quantized KV bytes are fed to both backends: the comparison is
                // purely between the two decoders (CPU f32 reference vs in-kernel MMA decode).
                const std::vector<float> q_src = gen_f32((size_t)(s.D*s.nq*s.nhq), 0xC0FFEEu);
                const std::vector<float> k_src = gen_f32((size_t)(s.D*NKV*s.nhkv), 0x1234u);
                const std::vector<float> v_src = gen_f32((size_t)(s.D*NKV*s.nhkv), 0x5678u);

                std::vector<uint8_t> k_bytes(kv_bytes);
                std::vector<uint8_t> v_bytes(kv_bytes);
                GGML_ASSERT(ggml_quantize_chunk(kv_type, k_src.data(), k_bytes.data(),
                                                0, NKV*s.nhkv, s.D, nullptr) == kv_bytes);
                GGML_ASSERT(ggml_quantize_chunk(kv_type, v_src.data(), v_bytes.data(),
                                                0, NKV*s.nhkv, s.D, nullptr) == kv_bytes);

                std::vector<uint8_t> mask_bytes;
                const std::vector<uint8_t> * mask_ptr = nullptr;
                if (s.mask) {
                    mask_bytes = gen_mask_f16(s.nq, NKV);
                    mask_ptr = &mask_bytes;
                }

                const std::vector<float> cpu_out = run_fa(
                    cpu_backend.get(), kv_type, s, q_src, k_bytes, v_bytes, mask_ptr);

                const uint64_t inv_before = ggml_backend_cuda_get_turbo_mma_fa_invocations();
                const std::vector<float> cuda_out = run_fa(
                    cuda_backend.get(), kv_type, s, q_src, k_bytes, v_bytes, mask_ptr);
                const uint64_t inv_delta =
                    ggml_backend_cuda_get_turbo_mma_fa_invocations() - inv_before;

                // 1. Dispatch sanity: the counter proves which kernel family ran.
                if (mma_mode) {
                    tt.assert_true("opt-in MMA dispatch counter advanced (GGML_CUDA_TURBO_MMA=1)",
                                   inv_delta >= 1);
                } else {
                    tt.assert_true("MMA dispatch counter stayed at 0 without the env var",
                                   inv_delta == 0);
                }

                // 2. Numerics vs the CPU reference decode of the same blocks.
                double max_out = 0.0;
                for (float f : cpu_out) {
                    max_out = std::max(max_out, std::abs((double) f));
                }
                double max_diff = 0.0;
                double sum_diff = 0.0;
                size_t arg = 0;
                for (size_t i = 0; i < cpu_out.size(); ++i) {
                    const double d = std::abs((double) cuda_out[i] - (double) cpu_out[i]);
                    sum_diff += d;
                    if (d > max_diff) {
                        max_diff = d;
                        arg = i;
                    }
                }
                const double mean_diff = sum_diff/(double) cpu_out.size();
                // The kernel decodes K/V into fp16 tiles (rel. rounding ~5e-4) and Q enters the
                // MMA as fp16 as well, so ~1e-3 relative is expected fp noise; a wrong tile or
                // rotation-group addressing bug would produce O(|out|) garbage instead.
                const double tol = 2e-2*max_out + 1e-4;
                printf("    invocations=%llu max_diff=%.3e (at %zu) mean_diff=%.3e max|out|=%.3f tol=%.3e\n",
                       (unsigned long long) inv_delta, max_diff, arg, mean_diff, max_out, tol);
                tt.assert_true("MMA output matches the CPU reference decode within fp16 tile noise",
                               max_diff <= tol);
            });
        }
    }

    return 0;
}
