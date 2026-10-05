// Differential test for the opt-in TurboQuant MMA FlashAttention path (Stage 1c L1,
// BUUN_MERGE_PLAN.md sections 1.4a / 10.9).
//
// What it verifies, per (K/V type pair, shape):
//   1. Dispatch: with GGML_CUDA_TURBO_MMA=1 every case below must increment the
//      ggml_backend_cuda_get_turbo_mma_fa_invocations() counter (i.e. the op really ran on
//      the non-f16 MMA kernel, not silently on the regular VEC/TILE/MMA_F16 path). Without the
//      env var ("fallback" mode) the counter must stay at 0. With the env var but no compiled
//      instance for the pair ("nopair" mode) it must also stay at 0 -- and the numeric check
//      must still pass, because 9l-d4b9w routes that case to the f16 tensor-core kernel rather
//      than letting it fall into VEC/TILE.
//   2. Numerics: the MMA kernel decodes the same quantized KV blocks the CPU reference
//      decodes, so the two outputs may only differ by in-kernel fp16 tile rounding and
//      accumulation order -- asserted to stay far below quantization-error scale, and far
//      below what a wrong tile/group addressing bug (O(1) garbage) would produce.
//
// The pairs cover every non-f16 MMA instantiation set (13 symmetric pairs: turbo scalar/TCQ,
// standard quantized, bf16; plus the mixed pairs); the shapes cover every tile the Volta
// dispatch ladder can select for D in {128, 256}: gqa 8/4/2/1 -> ncols2 8/4/2/1, small and
// large nq -> 32-column and 64-column rungs, and a no-mask case -> the ncols2=1 non-GQA path.
// No model file needed.
//
// Usage: test-turbo-mma-fa [fallback|nopair] [--only <substr>]
//   (default)  sets GGML_CUDA_TURBO_MMA=1 and asserts the MMA counter advances.
//   fallback   leaves the env var off and asserts the counter never advances; the numeric
//              comparison still runs against whatever regular kernel the backend picks.
//   nopair     sets the env var (so the opt-in is live) and asserts the counter stays at 0 for
//              every non-f16 pair: this is the 9l-d4b9w contract -- a pair with no compiled MMA
//              instance must be reported uncompiled and routed to the f16 tensor-core kernel,
//              not silently dropped into the VEC/TILE ladder. The numeric comparison must still
//              pass, which is the point: the fallback decodes correctly, it is not a no-op.
//              Only meaningful on a build where the pairs really are uncompiled (the default
//              GGML_CUDA_FA_MMA=OFF, or a FA_QUANTS list without them); the mode says so loudly
//              if a pair turns out to be compiled after all, because then it is measuring
//              nothing.
//   --only     runs only the cases whose name contains <substr> (fast device iteration).
//
// Each case reports nan(cpu/cuda) and sentinel_left; an output element still holding the
// pre-compute sentinel was never written by the kernel. Both are asserted to be 0 -- without
// them a NaN output would silently "pass" (every diff comparison against NaN is false, so a
// naive max_diff stays 0).

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

// Simple run statistics that make NaN/uninitialized outputs impossible to miss. A bare
// max-diff is blind here: any NaN element makes every comparison false, so max_diff stays 0
// and a completely unwritten/NaN output would look like a perfect match.
//   n_nan      : NaN elements in the output (must be 0)
//   n_sentinel : elements still holding the pre-compute sentinel, i.e. never written
struct run_stats {
    size_t n_nan      = 0;
    size_t n_sentinel = 0;
};

// Builds and runs one FLASH_ATTN_EXT on `backend`, returns the f32 output plus run_stats.
// `prefill_sentinel` (when not NaN) is written into the output tensor before the compute so
// that unwritten elements can be told apart from written NaN.
std::vector<float> run_fa(
        ggml_backend_t backend,
        ggml_type type_k,
        ggml_type type_v,
        const fa_shape & s,
        const std::vector<float> & q_src,
        const std::vector<uint8_t> & k_bytes,
        const std::vector<uint8_t> & v_bytes,
        const std::vector<uint8_t> * mask_bytes,
        float prefill_sentinel,
        run_stats * stats) {
    constexpr size_t N_TENSORS = 16;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false);
    const ggml_init_params params{context_bytes, nullptr, true};

    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,  s.D, s.nq, s.nhq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), type_k,         s.D, NKV, s.nhkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), type_v,         s.D, NKV, s.nhkv, 1);
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
    if (!std::isnan(prefill_sentinel)) {
        const std::vector<float> sentinel(ggml_nelements(kqvb), prefill_sentinel);
        ggml_backend_tensor_set(kqvb, sentinel.data(), 0, sentinel.size()*sizeof(float));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, kqvb);
    GGML_ASSERT(ggml_backend_supports_op(backend, kqvb));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> out(ggml_nelements(kqvb));
    GGML_ASSERT(ggml_is_contiguous(kqvb));
    ggml_backend_tensor_get(kqvb, out.data(), 0, out.size()*sizeof(float));

    if (stats != nullptr) {
        for (float f : out) {
            if (std::isnan(f) || std::isinf(f)) {
                ++stats->n_nan;
            } else if (!std::isnan(prefill_sentinel) && f == prefill_sentinel) {
                ++stats->n_sentinel;
            }
        }
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    testing t;

    // Three modes, distinguished by (env var, expected counter):
    //   mma      env=1  counter >= 1 on non-f16 pairs  -- the opt-in reaches the turbo kernel
    //   fallback env=0  counter == 0 everywhere        -- the opt-in changes nothing by default
    //   nopair   env=1  counter == 0 everywhere        -- the opt-in is live but no pair is
    //                                                      compiled, so 9l-d4b9w must route to
    //                                                      the f16 tensor-core kernel instead
    //                                                      of falling into VEC/TILE
    // The first argv token selects the mode; anything else is treated as the default (mma).
    const std::string mode = argc > 1 ? std::string(argv[1]) : std::string("mma");
    const bool fallback_mode = mode == "fallback";
    const bool nopair_mode   = mode == "nopair";
    const bool mma_mode      = !fallback_mode && !nopair_mode;

    // fallback is the only mode that leaves the opt-in off; nopair deliberately sets it, that is
    // the whole difference between it and fallback.
    if (!fallback_mode) {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_TURBO_MMA", "1");
#else
        setenv("GGML_CUDA_TURBO_MMA", "1", 1);
#endif
    }
    printf("mode: %s (GGML_CUDA_TURBO_MMA=%s)\n",
           fallback_mode ? "fallback" : nopair_mode ? "nopair" : "mma",
           fallback_mode ? "unset" : "1");
    if (nopair_mode) {
        printf("  expecting: every non-f16 pair has NO compiled MMA instance, so the opt-in\n"
               "  routes to the f16 tensor-core kernel. If a pair reports a counter advance,\n"
               "  this build DOES have that instance compiled and the mode measures nothing\n"
               "  (use the default mode instead, or configure without that pair).\n");
    }

    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    if (!cuda_backend) {
        printf("no CUDA device available - skipping\n");
        return 0;
    }
    ggml_backend_ptr cpu_backend(ggml_backend_cpu_init());
    GGML_ASSERT(cpu_backend);

    // K/V type pairs: every non-f16 MMA pair instantiated by the current GGML_CUDA_FA_QUANTS
    // selection -- the 13 symmetric pairs (turbo scalar/TCQ, standard quantized, bf16) plus the
    // mixed pairs. This exercises every general-loader decode branch (turbo 128-element rotation
    // groups, q4_0/q4_1/q5_0/q5_1/q8_0 32-element blocks, bf16 pseudo-blocks) and the asymmetric
    // combinations, not only the original turbo pairs. All pairs are in GGML_CUDA_FA_QUANTS, so
    // the dispatch-counter assertions apply to every case.
    struct kv_pair {
        ggml_type k;
        ggml_type v;
    };
    const kv_pair pairs[] = {
        // f16 K/V goes through the original f16 loader: this is the control that separates
        // "test harness + MMA path" from "the general non-f16 tile loader".
        {GGML_TYPE_F16,       GGML_TYPE_F16},
        {GGML_TYPE_TURBO8_0,  GGML_TYPE_TURBO8_0},
        {GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ},
        {GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ},
        {GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ},
        {GGML_TYPE_TURBO4_0,  GGML_TYPE_TURBO4_0},
        {GGML_TYPE_TURBO3_0,  GGML_TYPE_TURBO3_0},
        {GGML_TYPE_TURBO2_0,  GGML_TYPE_TURBO2_0},
        {GGML_TYPE_Q8_0,      GGML_TYPE_Q8_0},
        {GGML_TYPE_Q5_0,      GGML_TYPE_Q5_0},
        {GGML_TYPE_Q5_1,      GGML_TYPE_Q5_1},
        {GGML_TYPE_Q4_0,      GGML_TYPE_Q4_0},
        {GGML_TYPE_Q4_1,      GGML_TYPE_Q4_1},
        {GGML_TYPE_BF16,      GGML_TYPE_BF16},
        {GGML_TYPE_TURBO8_0,  GGML_TYPE_TURBO3_TCQ},
        {GGML_TYPE_F16,       GGML_TYPE_TURBO8_0},
        {GGML_TYPE_TURBO8_0,  GGML_TYPE_Q5_1},
        {GGML_TYPE_TURBO8_0,  GGML_TYPE_Q4_1},
        {GGML_TYPE_Q5_1,      GGML_TYPE_TURBO3_TCQ},
        {GGML_TYPE_Q4_1,      GGML_TYPE_TURBO3_TCQ},
        {GGML_TYPE_Q5_1,      GGML_TYPE_Q4_1},
        {GGML_TYPE_TURBO8_0,  GGML_TYPE_TURBO4_0},
        {GGML_TYPE_Q8_0,      GGML_TYPE_TURBO4_0},
        {GGML_TYPE_F16,       GGML_TYPE_Q8_0},
    };

    // Optional case filter for fast iteration on the device: --only <substr> matches the case
    // name (e.g. --only "turbo8-turbo8 D=128" or --only q5_0).
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--only" && i + 1 < argc) {
            only = argv[i + 1];
        }
    }

    for (const kv_pair & p : pairs) {
        for (const fa_shape & s : shapes) {
            std::string name = std::string(ggml_type_name(p.k)) + "-" +
                ggml_type_name(p.v) + " " + s.label;
            if (!only.empty() && name.find(only) == std::string::npos) {
                continue;
            }
            t.test(name.c_str(), [&](testing & tt) {
                const size_t k_row_bytes = ggml_row_size(p.k, s.D);
                const size_t v_row_bytes = ggml_row_size(p.v, s.D);
                const size_t k_bytes_len = k_row_bytes*(size_t)(NKV*s.nhkv);
                const size_t v_bytes_len = v_row_bytes*(size_t)(NKV*s.nhkv);

                // Identical quantized KV bytes are fed to both backends: the comparison is
                // purely between the two decoders (CPU f32 reference vs in-kernel MMA decode).
                const std::vector<float> q_src = gen_f32((size_t)(s.D*s.nq*s.nhq), 0xC0FFEEu);
                const std::vector<float> k_src = gen_f32((size_t)(s.D*NKV*s.nhkv), 0x1234u);
                const std::vector<float> v_src = gen_f32((size_t)(s.D*NKV*s.nhkv), 0x5678u);

                std::vector<uint8_t> k_bytes(k_bytes_len);
                std::vector<uint8_t> v_bytes(v_bytes_len);
                GGML_ASSERT(ggml_quantize_chunk(p.k, k_src.data(), k_bytes.data(),
                                                0, NKV*s.nhkv, s.D, nullptr) == k_bytes_len);
                GGML_ASSERT(ggml_quantize_chunk(p.v, v_src.data(), v_bytes.data(),
                                                0, NKV*s.nhkv, s.D, nullptr) == v_bytes_len);

                std::vector<uint8_t> mask_bytes;
                const std::vector<uint8_t> * mask_ptr = nullptr;
                if (s.mask) {
                    mask_bytes = gen_mask_f16(s.nq, NKV);
                    mask_ptr = &mask_bytes;
                }

                run_stats cpu_stats;
                const std::vector<float> cpu_out = run_fa(
                    cpu_backend.get(), p.k, p.v, s, q_src, k_bytes, v_bytes, mask_ptr,
                    std::nanf(""), &cpu_stats);

                // Sentinel: 1234.5 is not a value the attention of our inputs can produce, so any
                // element that still holds it after the compute was never written by the kernel.
                constexpr float SENTINEL = 1234.5f;
                run_stats cuda_stats;
                const uint64_t inv_before = ggml_backend_cuda_get_turbo_mma_fa_invocations();
                const std::vector<float> cuda_out = run_fa(
                    cuda_backend.get(), p.k, p.v, s, q_src, k_bytes, v_bytes, mask_ptr,
                    SENTINEL, &cuda_stats);
                const uint64_t inv_delta =
                    ggml_backend_cuda_get_turbo_mma_fa_invocations() - inv_before;

                // 1. Dispatch sanity: the counter proves which kernel family ran.
                //    f16 K/V is the control - it goes through the original f16 loader, so the
                //    TurboQuant MMA path must NOT claim it. Every other pair must be claimed,
                //    which is what separates "harness + MMA path" from the non-f16 tile loader.
                const bool expect_turbo_mma = !(p.k == GGML_TYPE_F16 && p.v == GGML_TYPE_F16);
                if (mma_mode) {
                    if (expect_turbo_mma) {
                        tt.assert_true("opt-in MMA dispatch counter advanced (GGML_CUDA_TURBO_MMA=1)",
                                       inv_delta >= 1);
                        if (inv_delta == 0) {
                            printf("    (MMA path not selected: is \"%s-%s\" in this build's "
                                   "GGML_CUDA_FA_QUANTS, and is the device Volta?)\n",
                                   ggml_type_name(p.k), ggml_type_name(p.v));
                        }
                    } else {
                        tt.assert_true("control pair f16/f16 keeps the original f16 loader",
                                       inv_delta == 0);
                    }
                } else if (nopair_mode) {
                    // 9l-d4b9w: the opt-in is live, so the pair is *not* skipped for lack of an
                    // env var -- it lacks an instance. The counter must therefore stay at 0, and
                    // the numeric comparison below is the load-bearing half: it proves the
                    // f16-MMA fallback actually decodes the same blocks correctly.
                    //
                    // A counter advance here is not a kernel bug, it means the premise is false
                    // (this pair IS compiled in this build), so say that instead of leaving the
                    // reader to guess from a bare "expected 0, got 1".
                    tt.assert_true(
                        "opt-in with an uncompiled pair does NOT reach the turbo MMA kernel",
                        inv_delta == 0);
                    if (inv_delta != 0) {
                        printf("    (nopair mode is void for this pair: \"%s-%s\" has a compiled "
                               "MMA instance in this build -- counter advanced. Use the default "
                               "mode to test the turbo kernel.)\n",
                               ggml_type_name(p.k), ggml_type_name(p.v));
                    } else {
                        printf("    (uncompiled pair routed away from the turbo MMA kernel as "
                               "expected; the numeric check below is what proves the fallback "
                               "decodes correctly)\n");
                    }
                } else {
                    tt.assert_true("MMA dispatch counter stayed at 0 without the env var",
                                   inv_delta == 0);
                }

                // 2. Output sanity: NaN/Inf and unwritten elements must never survive a compute.
                //    Without this, a NaN output makes every diff comparison false and the case
                //    would pass with max_diff == 0 (a false PASS).
                tt.assert_true("CPU reference output is finite", cpu_stats.n_nan == 0);
                tt.assert_true("CUDA output is finite (no NaN/Inf)", cuda_stats.n_nan == 0);
                tt.assert_true("CUDA output was fully written (no sentinel left)",
                               cuda_stats.n_sentinel == 0);

                // 3. Numerics vs the CPU reference decode of the same blocks.
                double max_out = 0.0;
                for (float f : cpu_out) {
                    if (std::isfinite(f)) {
                        max_out = std::max(max_out, std::abs((double) f));
                    }
                }
                double max_diff = 0.0;
                double sum_diff = 0.0;
                size_t arg = 0;
                size_t n_cmp = 0;
                for (size_t i = 0; i < cpu_out.size(); ++i) {
                    if (!std::isfinite(cpu_out[i]) || !std::isfinite(cuda_out[i])) {
                        continue; // reported by the finiteness assertions above
                    }
                    const double d = std::abs((double) cuda_out[i] - (double) cpu_out[i]);
                    sum_diff += d;
                    ++n_cmp;
                    if (d > max_diff) {
                        max_diff = d;
                        arg = i;
                    }
                }
                const double mean_diff = n_cmp > 0 ? sum_diff/(double) n_cmp : 0.0;
                // The kernel decodes K/V into fp16 tiles (rel. rounding ~5e-4) and Q enters the
                // MMA as fp16 as well, so ~1e-3 relative is expected fp noise; a wrong tile or
                // rotation-group addressing bug would produce O(|out|) garbage instead.
                const double tol = 2e-2*max_out + 1e-4;
                printf("    invocations=%llu nan(cpu/cuda)=%zu/%zu sentinel_left=%zu "
                       "max_diff=%.3e (at %zu) mean_diff=%.3e max|out|=%.3f tol=%.3e\n",
                       (unsigned long long) inv_delta, cpu_stats.n_nan, cuda_stats.n_nan,
                       cuda_stats.n_sentinel, max_diff, arg, mean_diff, max_out, tol);
                tt.assert_true("MMA output matches the CPU reference decode within fp16 tile noise",
                               max_diff <= tol && cuda_stats.n_nan == 0 && cuda_stats.n_sentinel == 0);
            });
        }
    }

    return 0;
}
