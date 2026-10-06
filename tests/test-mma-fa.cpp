// Differential test for the opt-in TurboQuant MMA FlashAttention path (Stage 1c L1,
// BUUN_MERGE_PLAN.md sections 1.4a / 10.9).
//
// What it verifies, per (K/V type pair, shape):
//   1. Routing: WHICH kernel family served the op -- read as a delta of
//      ggml_backend_cuda_get_fattn_kernel_invocations() around the compute, so the test names the
//      kernel that actually ran instead of inferring it from an absence. A claim of the form "the
//      opt-in must route this pair to X rather than Y" needs that: the turbo counter reads 0 for
//      every non-turbo family, and every one of those families decodes within tolerance, so
//      "counter stayed 0" cannot tell a correct fallback from the VEC/TILE drop that 9l-d4b9w
//      removed. Two further queries make the mode's premise checkable rather than assumed:
//      ggml_backend_cuda_fattn_mma_turbo_pair_compiled() (did this build compile the pair?) and
//      ggml_backend_cuda_fattn_mma_turbo_fallback_applies() (is the pair inside the fallback's
//      scope?). Both are answered by the backend, so the test cannot drift from the behaviour.
//   2. Numerics: the MMA kernel decodes the same quantized KV blocks the CPU reference
//      decodes, so the two outputs may only differ by in-kernel fp16 tile rounding and
//      accumulation order -- asserted to stay far below quantization-error scale, and far
//      below what a wrong tile/group addressing bug (O(1) garbage) would produce.
//
// The pairs cover every non-f16 MMA instantiation set (turbo scalar/TCQ, standard quantized,
// bf16; plus the mixed pairs) and all 17 combinations in the default GGML_CUDA_FA_QUANTS
// selection. The shapes cover every rung the Volta dispatch ladder can select for D in {128, 256}:
// gqa 8/4/2/1 -> ncols2 8/4/2/1, small and large nq -> 32-column and 64-column rungs, and a
// no-mask case -> the ncols2=1 non-GQA path. No model file needed.
//
// One shape deliberately reaches the VECTOR rung (fattn.cu: `nq * gqa_ratio_eff <= 2`): without
// it, every shape here has nq * gqa_ratio_eff >= 16, the ladder never selects VEC, and the
// "the opt-in must not drop this pair into VEC" half of the routing assertion is true by
// construction instead of by measurement. That is the whole confound 9l-d4b9w removes -- a turbo
// pair that is missing its MMA instance used to be served by the vector kernel on exactly these
// small-query shapes -- so the assertion has to be able to see them.
//
// Usage: test-mma-fa [fallback|nopair] [--only <substr>]
//   (default)  sets GGML_CUDA_FA_MMA=1, then holds each pair to the contract its build state
//              implies: a pair this build compiled must be dispatched to the turbo kernel, and a
//              pair it did not compile must NOT be claimed by the opt-in -- 9l-d4b9w has to land
//              it on the f16 tensor-core kernel, not on VEC/TILE. Green on any build config,
//              because "this pair was never compiled here" is a fact about the build, not a
//              defect, and the counter-only assertion used to report it as 184 failures.
//   nopair     sets the env var AND requires the stronger premise that this build compiled NO
//              pair. That is the only cell where the fallback is the sole thing that can serve the
//              pair, so the "the op ran on the f16 tensor-core kernel, and on neither VEC nor
//              TILE" assertion is 9l-d4b9w's proof rather than a restatement of "nothing ran".
//              A pair that turns out to be compiled makes the mode void for it -- reported as a
//              void premise, never as a kernel bug.
//   fallback   leaves the env var off and REQUIRES that it really is off, by reading it back. An
//              exported GGML_CUDA_FA_MMA would otherwise turn this mode into a silent duplicate
//              of nopair while every one of its `== 0` assertions still reported PASS.
//   --only     runs only the cases whose name contains <substr> (fast device iteration).
//
// Every case prints the kernel family that served it (kernel=...), so a log says which path ran
// rather than leaving the reader to deduce it. Each case also reports nan(cpu/cuda) and
// sentinel_left; an output element still holding the pre-compute sentinel was never written by
// the kernel. Both are asserted to be 0 -- without them a NaN output would silently "pass" (every
// diff comparison against NaN is false, so a naive max_diff stays 0).

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
    // The VEC rung: nq * gqa_ratio_eff == 2 is the ladder's only VEC gate (fattn.cu's
    // `can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2`), so this is the one shape on
    // which "opt-in on" and "opt-in off" differ by VEC vs MMA_F16 rather than TILE vs MMA_F16.
    // It is what keeps the VEC half of the routing assertion from being vacuous.
    {128,  2, 2, 2, true,  "D=128 gqa=1 nq=2  vec"},
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

// Which FA kernel family served an op. The family counters are the only observable that can name
// the kernel, so they are read as a delta around the compute, like the turbo counter.
struct fa_kernel_counters {
    uint64_t by_family[GGML_CUDA_FATTN_KERNEL_COUNT];
    uint64_t turbo;

    static fa_kernel_counters read() {
        fa_kernel_counters c{};
        for (int k = 0; k < GGML_CUDA_FATTN_KERNEL_COUNT; ++k) {
            c.by_family[k] = ggml_backend_cuda_get_fattn_kernel_invocations(k);
        }
        c.turbo = ggml_backend_cuda_get_turbo_mma_fa_invocations();
        return c;
    }

    fa_kernel_counters delta(const fa_kernel_counters & before) const {
        fa_kernel_counters d{};
        for (int k = 0; k < GGML_CUDA_FATTN_KERNEL_COUNT; ++k) {
            d.by_family[k] = by_family[k] - before.by_family[k];
        }
        d.turbo = turbo - before.turbo;
        return d;
    }

    // The one family this op was dispatched to, or -1 when the deltas do not describe exactly
    // one op. One FLASH_ATTN_EXT is one launch, so anything else means the probe, not the kernel.
    int which() const {
        int found = -1;
        for (int k = 0; k < GGML_CUDA_FATTN_KERNEL_COUNT; ++k) {
            if (by_family[k] == 0) {
                continue;
            }
            if (found >= 0 || by_family[k] != 1) {
                return -1;
            }
            found = k;
        }
        return found;
    }
};

const char * fattn_kernel_name(const int k) {
    switch (k) {
        case GGML_CUDA_FATTN_KERNEL_NONE:        return "none";
        case GGML_CUDA_FATTN_KERNEL_VEC:         return "vec";
        case GGML_CUDA_FATTN_KERNEL_TILE:        return "tile";
        case GGML_CUDA_FATTN_KERNEL_MMA_F16:     return "mma_f16";
        case GGML_CUDA_FATTN_KERNEL_MMA_TURBO:   return "mma_turbo";
        case GGML_CUDA_FATTN_KERNEL_VOLTA_Q8_W4: return "volta_q8_w4";
        default:                                 return "none-or-mixed";
    }
}

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

    // Three modes, distinguished by (env var, expected routing):
    //   mma      env=1  compiled pair -> turbo kernel; uncompiled pair -> the fallback's target
    //   fallback env=0  no family may be the turbo kernel
    //   nopair   env=1  AND no pair compiled: every in-scope pair must land on the f16
    //                   tensor-core kernel, and on neither VEC nor TILE -- 9l-d4b9w's proof
    // The first argv token selects the mode; anything else is treated as the default (mma).
    const std::string mode = argc > 1 ? std::string(argv[1]) : std::string("mma");
    const bool fallback_mode = mode == "fallback";
    const bool nopair_mode   = mode == "nopair";
    const bool mma_mode      = !fallback_mode && !nopair_mode;

    // fallback is the only mode that leaves the opt-in off; nopair deliberately sets it, that is
    // the whole difference between it and fallback.
    if (!fallback_mode) {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_FA_MMA", "1");
#else
        setenv("GGML_CUDA_FA_MMA", "1", 1);
#endif
    }

    // fallback's premise is an ABSENCE, so it has to be asserted, not merely not-created. A
    // GGML_CUDA_FA_MMA exported by the shell turns this mode into a copy of nopair, and every
    // assertion in it is of the `== 0` form, so it would keep reporting PASS while measuring
    // something else. Refuse to run instead of printing a claim that is not true.
    const char * env_now = std::getenv("GGML_CUDA_FA_MMA");
    const bool env_on = env_now != nullptr && std::atoi(env_now) != 0;
    if (fallback_mode && env_on) {
        printf("FATAL: GGML_CUDA_FA_MMA=%s is set in the environment, but this mode's whole\n"
               "  premise is that the opt-in is OFF. With it on, this run degenerates into nopair\n"
               "  and any PASS below would be measuring the wrong cell. Clear it and re-run:\n"
               "    set GGML_CUDA_FA_MMA=\n", env_now);
        return 2;
    }
    if (!fallback_mode && !env_on) {
        printf("FATAL: failed to set GGML_CUDA_FA_MMA for this mode.\n");
        return 2;
    }
    printf("mode: %s (GGML_CUDA_FA_MMA=%s)\n",
           fallback_mode ? "fallback" : nopair_mode ? "nopair" : "mma",
           env_now != nullptr ? env_now : "unset");
    if (nopair_mode) {
        printf("  expecting: this build compiled NO turbo MMA pair, so the opt-in must report each\n"
               "  one uncompiled and 9l-d4b9w must route it to the f16 tensor-core kernel -- which\n"
               "  is asserted by name below. A pair that turns out to be compiled makes this mode\n"
               "  void for that pair (reported as such, not as a kernel bug).\n");
    }

    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    if (!cuda_backend) {
        printf("no CUDA device available - skipping\n");
        return 0;
    }
    ggml_backend_ptr cpu_backend(ggml_backend_cpu_init());
    GGML_ASSERT(cpu_backend);

    // K/V type pairs: the symmetric pairs (turbo scalar/TCQ, standard quantized, bf16) plus the
    // mixed pairs, covering every non-f16 MMA instantiation set and all 17 combinations in
    // the default GGML_CUDA_FA_QUANTS list. This exercises every general-loader decode branch
    // (turbo 128-element rotation groups, q4_0/q4_1/q5_0/q5_1/q8_0 32-element blocks, bf16
    // pseudo-blocks) and the asymmetric combinations, not only the original turbo pairs. Whether
    // a pair is *compiled* in a given build is a separate question, asked per pair below -- it is
    // not assumed, because the default list, GGML_CUDA_FA_MMA and the configure cache can each
    // disagree with it.
    struct kv_pair {
        ggml_type k;
        ggml_type v;
    };
    const kv_pair pairs[] = {
        // f16 K/V goes through the original f16 loader: this is the control that separates
        // "test harness + MMA path" from "the general non-f16 tile loader". It is also outside
        // the opt-in's fallback scope, so the control arm below is derived from the predicate
        // rather than from "everything except f16".
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
        {GGML_TYPE_TURBO4_0,  GGML_TYPE_TURBO3_TCQ},
        {GGML_TYPE_Q8_0,      GGML_TYPE_TURBO4_0},
        {GGML_TYPE_F16,       GGML_TYPE_Q8_0},
        // The three default-list combinations the table above does not reach. Each pairs a
        // standard-quant or f16 K with a turbo4_0 / q5_1 V, and no VBR ladder produces any of them
        // (fa_ladder_coverage.py shows the turbo and classic closures contain none), yet they are
        // legal *explicit* selections: kv_cache_types accepts Q4_1/Q5_1/TURBO4_0 as a -ctk/-ctv
        // value, so `-ctk q4_1 -ctv turbo4_0` parses, and with both sides pinned it is a static
        // cache the VBR controller never touches. FA_QUANTS exists to instantiate exactly those
        // build-time combinations, and the tree carries all of it: a vec instance AND its
        // FATTN_VEC_CASES_ALL_D dispatch entry (so VEC is a real candidate for these, which is
        // what makes nopair's "VEC == 0" assertion mean something), 16 MMA tiles, and the
        // pair_compiled + typed-dispatch guard entries. K is the finer side in all three
        // (f16/q5_1 = 16.0/6.0, q5_1/t4 = 6.0/4.125, q4_1/t4 = 5.0/4.125 bpv), so they exercise
        // 9l-d4b9z's K>=V rule from the allowed direction. Before this they had no coverage.
        {GGML_TYPE_F16,       GGML_TYPE_Q5_1},
        {GGML_TYPE_Q5_1,      GGML_TYPE_TURBO4_0},
        {GGML_TYPE_Q4_1,      GGML_TYPE_TURBO4_0},
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
        // Two build/scope facts the mode's contract is written in terms of. Asking the backend
        // (rather than re-deriving them here) is what keeps the test honest: these are the same
        // predicates the dispatch uses, so a PASS here means what it says.
        const bool pair_compiled = ggml_backend_cuda_fattn_mma_turbo_pair_compiled(p.k, p.v);
        const bool in_scope      = ggml_backend_cuda_fattn_mma_turbo_fallback_applies(p.k, p.v);
        const char * name_k = ggml_type_name(p.k);
        const char * name_v = ggml_type_name(p.v);

        for (const fa_shape & s : shapes) {
            std::string name = std::string(name_k) + "-" + name_v + " " + s.label;
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
                const fa_kernel_counters kc_before = fa_kernel_counters::read();
                const std::vector<float> cuda_out = run_fa(
                    cuda_backend.get(), p.k, p.v, s, q_src, k_bytes, v_bytes, mask_ptr,
                    SENTINEL, &cuda_stats);
                const fa_kernel_counters kc = fa_kernel_counters::read().delta(kc_before);
                const int  ran       = kc.which();
                const uint64_t inv_delta = kc.turbo;

                // 1. Routing: name the family that served this case, then hold it to what this
                //    mode can legitimately require. "Compiled here?" and "in the fallback's
                //    scope?" come from the backend, so each arm below states a fact, not a guess.
                if (mma_mode) {
                    if (pair_compiled) {
                        tt.assert_true("compiled pair is dispatched to the turbo MMA kernel",
                                       ran == GGML_CUDA_FATTN_KERNEL_MMA_TURBO);
                        if (ran != GGML_CUDA_FATTN_KERNEL_MMA_TURBO) {
                            printf("    (this build reports \"%s-%s\" as COMPILED, yet the dispatch "
                                   "chose %s. That is a routing defect -- the guard tables and the "
                                   "compiled instances disagree -- not a build-configuration "
                                   "question.)\n", name_k, name_v, fattn_kernel_name(ran));
                        }
                    } else {
                        tt.assert_true("the opt-in does not claim a pair this build did not compile",
                                       ran != GGML_CUDA_FATTN_KERNEL_MMA_TURBO);
                        if (in_scope) {
                            // 9l-d4b9w checked in the default mode too: with the opt-in live and the
                            // instance missing, "tensor cores, whatever it takes" means MMA_F16.
                            tt.assert_true("9l-d4b9w: uncompiled pair lands on the f16 tensor-core kernel",
                                           ran == GGML_CUDA_FATTN_KERNEL_MMA_F16);
                            tt.assert_true("9l-d4b9w: uncompiled pair is NOT dropped into VEC/TILE",
                                           kc.by_family[GGML_CUDA_FATTN_KERNEL_VEC] == 0 &&
                                           kc.by_family[GGML_CUDA_FATTN_KERNEL_TILE] == 0);
                            if (ran != GGML_CUDA_FATTN_KERNEL_MMA_F16) {
                                printf("    (9l-d4b9w: \"%s-%s\" is uncompiled in this build and in the "
                                       "fallback's scope, so MMA_F16 was required -- but %s served it.)\n",
                                       name_k, name_v, fattn_kernel_name(ran));
                            }
                        } else {
                            printf("    (control: \"%s-%s\" is outside the opt-in's fallback scope "
                                   "(plain f16 KV), so the opt-in deliberately changes nothing and the "
                                   "ordinary selection stands -- %s)\n",
                                   name_k, name_v, fattn_kernel_name(ran));
                        }
                    }
                } else if (nopair_mode) {
                    // 9l-d4b9w: the opt-in is live, so the pair is *not* skipped for lack of an
                    // env var -- it lacks an instance. The counter must stay at 0, and the family
                    // must be the f16 tensor-core kernel: that pair of facts is the fallback's
                    // proof, and it is only available in this mode because only here is the
                    // fallback the sole candidate.
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
                               "mode to test the turbo kernel.)\n", name_k, name_v);
                    } else if (in_scope) {
                        tt.assert_true("9l-d4b9w: the uncompiled pair IS routed to the f16 tensor-core kernel",
                                       ran == GGML_CUDA_FATTN_KERNEL_MMA_F16);
                        tt.assert_true("9l-d4b9w: the uncompiled pair is NOT dropped into VEC/TILE",
                                       kc.by_family[GGML_CUDA_FATTN_KERNEL_VEC] == 0 &&
                                       kc.by_family[GGML_CUDA_FATTN_KERNEL_TILE] == 0);
                        if (ran != GGML_CUDA_FATTN_KERNEL_MMA_F16) {
                            printf("    (9l-d4b9w: \"%s-%s\" is uncompiled and in the fallback's scope, "
                                   "so the f16 tensor-core kernel was required -- but %s served it. "
                                   "Before 9l-d4b9w this is exactly what the VEC/TILE ladder did.)\n",
                                   name_k, name_v, fattn_kernel_name(ran));
                        }
                    } else {
                        printf("    (control: \"%s-%s\" is outside the opt-in's fallback scope "
                               "(plain f16 KV) and has no turbo instance to begin with, so this pair "
                               "cannot witness the fallback; the ordinary selection stands -- %s)\n",
                               name_k, name_v, fattn_kernel_name(ran));
                    }
                } else {
                    tt.assert_true("MMA dispatch counter stayed at 0 without the env var",
                                   inv_delta == 0);
                    tt.assert_true("no turbo MMA launch without the env var",
                                   ran != GGML_CUDA_FATTN_KERNEL_MMA_TURBO);
                    if (pair_compiled) {
                        printf("    (the opt-in is off, so this pair's compiled turbo instance is "
                               "deliberately not selected -- %s served it)\n", fattn_kernel_name(ran));
                    }
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
                printf("    kernel=%-9s compiled=%d in_scope=%d invocations=%llu nan(cpu/cuda)=%zu/%zu "
                       "sentinel_left=%zu max_diff=%.3e (at %zu) mean_diff=%.3e max|out|=%.3f tol=%.3e\n",
                       fattn_kernel_name(ran), (int) pair_compiled, (int) in_scope,
                       (unsigned long long) inv_delta, cpu_stats.n_nan, cuda_stats.n_nan,
                       cuda_stats.n_sentinel, max_diff, arg, mean_diff, max_out, tol);
                tt.assert_true("MMA output matches the CPU reference decode within fp16 tile noise",
                               max_diff <= tol && cuda_stats.n_nan == 0 && cuda_stats.n_sentinel == 0);
            });
        }
    }

    // Exit code and a summary with the failure counts: without this the process returned 0 no
    // matter how many assertions failed, so a ctest/CI invocation could not see a red run at all.
    return t.summary();
}
