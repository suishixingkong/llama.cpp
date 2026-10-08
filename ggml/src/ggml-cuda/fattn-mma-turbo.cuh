// TurboQuant / general quantized KV-cache tensor-core FlashAttention launcher
// (MERGE_BUUN_PLAN.md sections 1.4a / 10.9 / 10.10).
//
// Stage 1c L1: the MMA FA kernel body (fattn-mma-f16.cuh) was made type-generic (type_K/type_V
// threaded through flash_attn_ext_f16/_process_tile/_iter, with flash_attn_ext_turbo_load_tile
// selected for non-f16 KV tiles). This header adds the missing top half:
//   - a shared-memory size helper that reflects the forced-synchronous (nstages == 0) pipeline
//     the kernel runs for non-f16 KV (cp.async cannot dequantize in flight),
//   - the per-(DKQ, DV, ncols1, ncols2, type_K, type_V) launcher, which hands raw (non-converted)
//     K/V to launch_fattn(need_f16_K/V = false) so the kernel decodes blocks in place,
//   - the dispatch ladder, mirroring ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1/2 in
//     fattn.cu but restricted to tiles Volta can run (ncols1*ncols2 >= 32, no sparse variant),
//   - the (K,V) pair table: which pairs have instances, and the pair-resolving typed dispatch.
//
// Instance files: template-instances/fattn-mma-<type_K>-<type_V>-instance-dkq<D>-ncols1_<A>-ncols2_<B>.cu
// Which of them exist as source files follows the vec instances (GGML_CUDA_FA_QUANTS ->
// FA_COMBINATIONS), see ggml_cuda_fattn_mma_turbo_instances() in ggml/cmake/common.cmake. Whether
// any of them are *compiled* is a separate, stricter gate: GGML_CUDA_FA_MMA (CMake, default
// OFF). With that gate off the build contains no non-f16 MMA kernel at all, so every reference
// below is ANDed with GGML_CUDA_FA_MMA and an unreachable pair keeps the regular kernel
// selection instead of producing an undefined symbol at link time.
//
// The streamed span prefill (kv-stream) additionally uses the partial-output variant
// (ggml_cuda_flash_attn_ext_mma_turbo_partial_case) for the (D=256, tile 8x8) shape so that
// DIRECT turbo pairs skip the f16 conversion entirely; see the -partial instance files.

#pragma once

#include "common.cuh"
#include "fattn-mma-f16.cuh"

// Master switch for the non-f16 MMA kernels. CMake (ggml_fattn_mma_quant_instances() in
// ggml/cmake/common.cmake) defines it as 1 when GGML_CUDA_FA_MMA is ON and leaves it
// undefined otherwise, which this block turns into 0. It is ANDed into every per-pair guard below
// because GGML_CUDA_FA_<K>_<V> alone is not evidence that a pair's MMA instance was compiled:
// that macro also gates the VEC instances and stays 1 for every GGML_CUDA_FA_QUANTS pair even when
// the MMA instances were switched off. Defaulting to 0 also keeps a translation unit compiled
// outside CMake from referencing an instance that does not exist.
#ifndef GGML_CUDA_FA_MMA
#define GGML_CUDA_FA_MMA 0
#endif

// Dynamic shared memory the MMA FA kernel requests for a non-f16 (DKQ, DV, ncols1, ncols2)
// combination on this arch. Same formula as ggml_cuda_fattn_mma_f16_shared_bytes(), but with
// nstages forced to 0: the kernel hardcodes the synchronous single-buffered pipeline for
// non-f16 KV (constexpr in flash_attn_ext_f16_process_tile/_iter), so the double-buffered
// cp.async KV staging that the f16 helper accounts for never exists here. Host-side consumers:
// the launcher below (cudaFuncSetAttribute) and the dispatch in fattn.cu (tile selection).
static size_t ggml_cuda_fattn_mma_turbo_shared_bytes(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    const int ncols          = ncols1*ncols2;
    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa_layout(DKQ, DV, ncols1, ncols2, cc);
    const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    const bool V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    const int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_tile_V = V_is_K_view ? stride_tile_K : ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);

    const size_t nbytes_shared_KV = nbatch_fa * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q  = ncols  * (DKQ/2 + 4)               * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4)       * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps*cols_per_warp * (nbatch_combine + 4) * sizeof(half2);

    return std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);
}

// Launcher for one (DKQ, DV, ncols1, ncols2, type_K, type_V) non-f16 MMA case. Explicitly
// instantiated by the fattn-mma-<k>-<v>-instance-*.cu files; the dispatch in fattn.cu only
// references these instantiations (via the pair table below) when the pair's
// GGML_CUDA_FA_<K>_<V> macro says the instance was compiled, so there is never a dangling
// reference for pairs that GGML_CUDA_FA_QUANTS left out.
template<int DKQ, int DV, int ncols1, int ncols2, bool output_partial, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_case_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        float * partial_dst, float2 * partial_meta) {
    static_assert(type_K != GGML_TYPE_F16 || type_V != GGML_TYPE_F16,
        "the general MMA tile launcher is for non-f16 KV; use ggml_cuda_flash_attn_ext_mma_f16_case for f16 KV");

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa_layout(DKQ, DV, ncols1, ncols2, cc);
    const int warp_size_host = ggml_cuda_info().devices[id].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    static_assert(DKQ != 576, "no general-tile MMA instances for Deepseek-style V-is-K-view shapes");

    const size_t nbytes_shared_total = ggml_cuda_fattn_mma_turbo_shared_bytes(DKQ, DV, ncols1, ncols2, cc);

    // The dispatch above is supposed to pick a tile that fits this device's opt-in shared-memory
    // limit. Fail loudly here instead of surfacing an opaque cudaErrorInvalidValue from
    // cudaFuncSetAttribute below.
    if (nbytes_shared_total > ggml_cuda_info().devices[id].smpbo) {
        GGML_ABORT("fatal error: flash_attn_ext_f16<non-f16 %d,%d,%d,%d> needs %zu bytes of dynamic shared memory, "
                   "exceeding device %d's %zu-byte limit; the dispatch chose a tile too wide for this GPU\n",
                   DKQ, DV, ncols1, ncols2, nbytes_shared_total, id, ggml_cuda_info().devices[id].smpbo);
    }

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void *;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, false, output_partial, type_K, type_V>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, false, output_partial, type_K, type_V>;
    }

    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes_shared_total));
        shared_memory_limit_raised[id] = true;
    }

    // need_f16_K/V == false: launch_fattn passes the raw quantized K/V blocks through and the
    // kernel decodes them tile-by-tile via flash_attn_ext_turbo_load_tile.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false,
         !output_partial, false, warp_size_host, partial_dst, partial_meta);
}

template<int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_mma_turbo_case_impl<DKQ, DV, ncols1, ncols2, false, type_K, type_V>(
        ctx, dst, nullptr, nullptr);
}

template<int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_partial_case(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        float * partial_dst, float2 * partial_meta) {
    GGML_ASSERT(partial_dst != nullptr && partial_meta != nullptr);
    ggml_cuda_flash_attn_ext_mma_turbo_case_impl<DKQ, DV, ncols1, ncols2, true, type_K, type_V>(
        ctx, dst, partial_dst, partial_meta);
}

template<int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_partial_case(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        float * partial_dst, float2 * partial_meta);

#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, type_K, type_V)                        \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                         \
    <DKQ, DV, ncols1, ncols2, GGML_TYPE_##type_K, GGML_TYPE_##type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

#define DECL_FATTN_MMA_TURBO_PARTIAL_CASE(DKQ, DV, ncols1, ncols2, type_K, type_V)                \
    template void ggml_cuda_flash_attn_ext_mma_turbo_partial_case                                 \
    <DKQ, DV, ncols1, ncols2, GGML_TYPE_##type_K, GGML_TYPE_##type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float *, float2 *)

// The (K,V) pair table: pairs with MMA tile instances in this repository. A pair participates
// in the dispatch only when its GGML_CUDA_FA_<K>_<V> macro says the instances were compiled
// (same values that gate the vec instance compilation, set from GGML_CUDA_FA_QUANTS), so an
// uncompiled pair silently keeps the regular kernel selection instead of failing to link.
static bool ggml_cuda_fattn_mma_turbo_pair_compiled(ggml_type type_k, ggml_type type_v) {
#define TURBO_MMA_PAIR_TEST(tk, tv) \
    if (type_k == GGML_TYPE_##tk && type_v == GGML_TYPE_##tv) { return true; }
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO8_0) && GGML_CUDA_FA_TURBO8_0_TURBO8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO8_0, TURBO8_0)
#endif
#if defined(GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ) && GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO3_TCQ, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ) && GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO2_TCQ, TURBO2_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ) && GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO1_TCQ, TURBO1_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO8_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_F16_TURBO8_0) && GGML_CUDA_FA_F16_TURBO8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(F16, TURBO8_0)
#endif
#if defined(GGML_CUDA_FA_TURBO4_0_TURBO4_0) && GGML_CUDA_FA_TURBO4_0_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO4_0, TURBO4_0)
#endif
// fork 9l-d4b9y: t4 x TCQ. The VBR ladder reaches this when the K side has bottomed out at
// t4 (4.125 bpv) while the V side keeps degrading; K staying the finer side is what makes the
// pair reachable at all under the K>=V rule. No instance existed in either tree, so no
// GGML_CUDA_FA_QUANTS list could select it.
#if defined(GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO4_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO3_0_TURBO3_0) && GGML_CUDA_FA_TURBO3_0_TURBO3_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO3_0, TURBO3_0)
#endif
#if defined(GGML_CUDA_FA_TURBO2_0_TURBO2_0) && GGML_CUDA_FA_TURBO2_0_TURBO2_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO2_0, TURBO2_0)
#endif
#if defined(GGML_CUDA_FA_Q8_0_Q8_0) && GGML_CUDA_FA_Q8_0_Q8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q8_0, Q8_0)
#endif
#if defined(GGML_CUDA_FA_Q5_0_Q5_0) && GGML_CUDA_FA_Q5_0_Q5_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q5_0, Q5_0)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q5_1) && GGML_CUDA_FA_Q5_1_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q5_1, Q5_1)
#endif
#if defined(GGML_CUDA_FA_Q4_0_Q4_0) && GGML_CUDA_FA_Q4_0_Q4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q4_0, Q4_0)
#endif
#if defined(GGML_CUDA_FA_Q4_1_Q4_1) && GGML_CUDA_FA_Q4_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q4_1, Q4_1)
#endif
#if defined(GGML_CUDA_FA_BF16_BF16) && GGML_CUDA_FA_BF16_BF16 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(BF16, BF16)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q5_1) && GGML_CUDA_FA_TURBO8_0_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO8_0, Q5_1)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q4_1) && GGML_CUDA_FA_TURBO8_0_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO8_0, Q4_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_TURBO3_TCQ) && GGML_CUDA_FA_Q5_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q5_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q4_1_TURBO3_TCQ) && GGML_CUDA_FA_Q4_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q4_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q4_1) && GGML_CUDA_FA_Q5_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q5_1, Q4_1)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO4_0) && GGML_CUDA_FA_TURBO8_0_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(TURBO8_0, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_Q8_0_TURBO4_0) && GGML_CUDA_FA_Q8_0_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q8_0, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_F16_Q8_0) && GGML_CUDA_FA_F16_Q8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(F16, Q8_0)
#endif
#if defined(GGML_CUDA_FA_F16_Q5_1) && GGML_CUDA_FA_F16_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(F16, Q5_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_TURBO4_0) && GGML_CUDA_FA_Q5_1_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q5_1, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_Q4_1_TURBO4_0) && GGML_CUDA_FA_Q4_1_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_TEST(Q4_1, TURBO4_0)
#endif
#undef TURBO_MMA_PAIR_TEST
    return false;
}

// Instantiates the case for whichever compiled (K,V) pair the tensors hold. Pairs without a
// compiled instance never reach this (the selection in fattn.cu checks
// ggml_cuda_fattn_mma_turbo_pair_compiled first).
template <int DKQ, int DV, int ncols1, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_turbo_case_typed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type type_K = dst->src[1]->type;
    const ggml_type type_V = dst->src[2]->type;

#define TURBO_MMA_PAIR_CALL(tk, tv)                                                                     \
    if (type_K == GGML_TYPE_##tk && type_V == GGML_TYPE_##tv) {                                         \
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, ncols1, ncols2, GGML_TYPE_##tk, GGML_TYPE_##tv>(ctx, dst); \
        return;                                                                                         \
    }
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO8_0) && GGML_CUDA_FA_TURBO8_0_TURBO8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO8_0, TURBO8_0)
#endif
#if defined(GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ) && GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO3_TCQ, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ) && GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO2_TCQ, TURBO2_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ) && GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO1_TCQ, TURBO1_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO8_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_F16_TURBO8_0) && GGML_CUDA_FA_F16_TURBO8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(F16, TURBO8_0)
#endif
#if defined(GGML_CUDA_FA_TURBO4_0_TURBO4_0) && GGML_CUDA_FA_TURBO4_0_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO4_0, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO4_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO3_0_TURBO3_0) && GGML_CUDA_FA_TURBO3_0_TURBO3_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO3_0, TURBO3_0)
#endif
#if defined(GGML_CUDA_FA_TURBO2_0_TURBO2_0) && GGML_CUDA_FA_TURBO2_0_TURBO2_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO2_0, TURBO2_0)
#endif
#if defined(GGML_CUDA_FA_Q8_0_Q8_0) && GGML_CUDA_FA_Q8_0_Q8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q8_0, Q8_0)
#endif
#if defined(GGML_CUDA_FA_Q5_0_Q5_0) && GGML_CUDA_FA_Q5_0_Q5_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q5_0, Q5_0)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q5_1) && GGML_CUDA_FA_Q5_1_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q5_1, Q5_1)
#endif
#if defined(GGML_CUDA_FA_Q4_0_Q4_0) && GGML_CUDA_FA_Q4_0_Q4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q4_0, Q4_0)
#endif
#if defined(GGML_CUDA_FA_Q4_1_Q4_1) && GGML_CUDA_FA_Q4_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q4_1, Q4_1)
#endif
#if defined(GGML_CUDA_FA_BF16_BF16) && GGML_CUDA_FA_BF16_BF16 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(BF16, BF16)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q5_1) && GGML_CUDA_FA_TURBO8_0_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO8_0, Q5_1)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q4_1) && GGML_CUDA_FA_TURBO8_0_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO8_0, Q4_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_TURBO3_TCQ) && GGML_CUDA_FA_Q5_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q5_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q4_1_TURBO3_TCQ) && GGML_CUDA_FA_Q4_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q4_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q4_1) && GGML_CUDA_FA_Q5_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q5_1, Q4_1)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO4_0) && GGML_CUDA_FA_TURBO8_0_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(TURBO8_0, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_Q8_0_TURBO4_0) && GGML_CUDA_FA_Q8_0_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q8_0, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_F16_Q8_0) && GGML_CUDA_FA_F16_Q8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(F16, Q8_0)
#endif
#if defined(GGML_CUDA_FA_F16_Q5_1) && GGML_CUDA_FA_F16_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(F16, Q5_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_TURBO4_0) && GGML_CUDA_FA_Q5_1_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q5_1, TURBO4_0)
#endif
#if defined(GGML_CUDA_FA_Q4_1_TURBO4_0) && GGML_CUDA_FA_Q4_1_TURBO4_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PAIR_CALL(Q4_1, TURBO4_0)
#endif
#undef TURBO_MMA_PAIR_CALL
    GGML_ABORT("fatal error: no non-f16 MMA instance compiled for K/V types %s-%s (add the pair to GGML_CUDA_FA_QUANTS)",
               ggml_type_name(type_K), ggml_type_name(type_V));
}

// Typed partial dispatch for the streamed span prefill. Returns false (without doing anything)
// when the pair's partial instance was not compiled, so the caller can keep its f16 route.
template <int DKQ, int DV, int ncols1, int ncols2>
static bool ggml_cuda_flash_attn_ext_mma_turbo_partial_case_typed(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        float * partial_dst, float2 * partial_meta) {
    const ggml_type type_K = dst->src[1]->type;
    const ggml_type type_V = dst->src[2]->type;

#define TURBO_MMA_PARTIAL_CALL(tk, tv)                                                                  \
    if (type_K == GGML_TYPE_##tk && type_V == GGML_TYPE_##tv) {                                         \
        ggml_cuda_flash_attn_ext_mma_turbo_partial_case<DKQ, DV, ncols1, ncols2, GGML_TYPE_##tk, GGML_TYPE_##tv>(ctx, dst, partial_dst, partial_meta); \
        return true;                                                                                    \
    }
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO8_0) && GGML_CUDA_FA_TURBO8_0_TURBO8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO8_0, TURBO8_0)
#endif
#if defined(GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ) && GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO3_TCQ, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO4_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ) && GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO2_TCQ, TURBO2_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ) && GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO1_TCQ, TURBO1_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO8_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q5_1) && GGML_CUDA_FA_TURBO8_0_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO8_0, Q5_1)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q4_1) && GGML_CUDA_FA_TURBO8_0_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(TURBO8_0, Q4_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_TURBO3_TCQ) && GGML_CUDA_FA_Q5_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(Q5_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q4_1_TURBO3_TCQ) && GGML_CUDA_FA_Q4_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(Q4_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q5_1) && GGML_CUDA_FA_Q5_1_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(Q5_1, Q5_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q4_1) && GGML_CUDA_FA_Q5_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(Q5_1, Q4_1)
#endif
#if defined(GGML_CUDA_FA_Q4_1_Q4_1) && GGML_CUDA_FA_Q4_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_CALL(Q4_1, Q4_1)
#endif
#undef TURBO_MMA_PARTIAL_CALL
    return false;
}

// Whether the streamed span prefill can route this pair to a typed non-f16 MMA partial kernel
// (the partial instances exist for the D=256 / tile 8x8 shape of these pairs only). Pairs whose
// macro is 0 report false so the span keeps its f16-conversion route.
static bool ggml_cuda_flash_attn_ext_mma_turbo_partial_available(ggml_type type_k, ggml_type type_v) {
#define TURBO_MMA_PARTIAL_TEST(tk, tv) \
    if (type_k == GGML_TYPE_##tk && type_v == GGML_TYPE_##tv) { return true; }
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO8_0) && GGML_CUDA_FA_TURBO8_0_TURBO8_0 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO8_0, TURBO8_0)
#endif
#if defined(GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ) && GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO3_TCQ, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO4_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO4_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ) && GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO2_TCQ, TURBO2_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ) && GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO1_TCQ, TURBO1_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ) && GGML_CUDA_FA_TURBO8_0_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO8_0, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q5_1) && GGML_CUDA_FA_TURBO8_0_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO8_0, Q5_1)
#endif
#if defined(GGML_CUDA_FA_TURBO8_0_Q4_1) && GGML_CUDA_FA_TURBO8_0_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(TURBO8_0, Q4_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_TURBO3_TCQ) && GGML_CUDA_FA_Q5_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(Q5_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q4_1_TURBO3_TCQ) && GGML_CUDA_FA_Q4_1_TURBO3_TCQ && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(Q4_1, TURBO3_TCQ)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q5_1) && GGML_CUDA_FA_Q5_1_Q5_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(Q5_1, Q5_1)
#endif
#if defined(GGML_CUDA_FA_Q5_1_Q4_1) && GGML_CUDA_FA_Q5_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(Q5_1, Q4_1)
#endif
#if defined(GGML_CUDA_FA_Q4_1_Q4_1) && GGML_CUDA_FA_Q4_1_Q4_1 && GGML_CUDA_FA_MMA
    TURBO_MMA_PARTIAL_TEST(Q4_1, Q4_1)
#endif
#undef TURBO_MMA_PARTIAL_TEST
    return false;
}

// Volta-only ladder. Mirrors ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1 (fattn.cu) with the
// Volta-only restrictions applied unconditionally: no sparse variant, and no rung below the
// 32-column tile (the MMA kernel body compiles ncols1*ncols2 < 32 to NO_DEVICE_CODE on Volta).
template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case_typed<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    // Prefer the wider 64-column tile, but only when it fits the device's opt-in shared-memory
    // limit (sm_70 has only 96 KB). The helper (nstages == 0) sizes the synchronous
    // pipeline the kernel will actually run.
    if (32/ncols2 >= 1 &&
            ggml_cuda_fattn_mma_turbo_shared_bytes(DKQ, DV, 64/ncols2, ncols2, cc) >
                ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        ggml_cuda_flash_attn_ext_mma_turbo_case_typed<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_case_typed<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // Same ladder ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2 uses on Volta (DKQ <= 256):
    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

// Explicit instantiation DECLARATIONS for every (pair, shape) the dispatch can reach. Like the
// f16 DECL block above, the extern suppresses implicit instantiation in the including TUs
// (notably fattn.cu): the kernel code is emitted only by the fattn-mma-<k>-<v>-instance-*.cu
// files' non-extern DECL definitions, whose compilation is gated on GGML_CUDA_FA_QUANTS.
// Pairs whose macro is 0 never reference their declarations, so a missing instance is fine.
#define DECL_FATTN_MMA_TURBO_CASE_EXTERN(DKQ, DV, ncols1, ncols2, type_K, type_V) \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, type_K, type_V)

#define DECL_FATTN_MMA_TURBO_PAIR_SHAPES(type_K, type_V)             \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128, 32, 1, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128, 64, 1, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128, 16, 2, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128, 32, 2, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128,  8, 4, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128, 16, 4, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128,  4, 8, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(128, 128,  8, 8, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256, 32, 1, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256, 64, 1, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256, 16, 2, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256, 32, 2, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256,  8, 4, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256, 16, 4, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256,  4, 8, type_K, type_V); \
    DECL_FATTN_MMA_TURBO_CASE_EXTERN(256, 256,  8, 8, type_K, type_V);

DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO8_0,  TURBO8_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO3_TCQ, TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO2_TCQ, TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO1_TCQ, TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO8_0,  TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(F16,       TURBO8_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO4_0,  TURBO4_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO3_0,  TURBO3_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO2_0,  TURBO2_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q8_0,      Q8_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q5_0,      Q5_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q5_1,      Q5_1)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q4_0,      Q4_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q4_1,      Q4_1)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(BF16,      BF16)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO8_0,  Q5_1)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO8_0,  Q4_1)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q5_1,      TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q4_1,      TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(F16,       Q5_1)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q5_1,      TURBO4_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q4_1,      TURBO4_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q5_1,      Q4_1)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(TURBO8_0,  TURBO4_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(Q8_0,      TURBO4_0)
DECL_FATTN_MMA_TURBO_PAIR_SHAPES(F16,       Q8_0)

// Streamed span prefill uses exactly this shape (D=256, tile 8x8):
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO8_0,  TURBO8_0);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO3_TCQ, TURBO3_TCQ);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO2_TCQ, TURBO2_TCQ);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO1_TCQ, TURBO1_TCQ);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO8_0,  TURBO3_TCQ);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO8_0,  Q5_1);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, TURBO8_0,  Q4_1);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, Q5_1,      TURBO3_TCQ);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, Q4_1,      TURBO3_TCQ);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, Q5_1,      Q5_1);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, Q5_1,      Q4_1);
extern DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, Q4_1,      Q4_1);
