// TurboQuant KV-cache tensor-core FlashAttention launcher (BUUN_MERGE_PLAN.md section 1.4a).
//
// Stage 1c L1: the MMA FA kernel body (fattn-mma-f16.cuh) was made type-generic (type_K/type_V
// threaded through flash_attn_ext_f16/_process_tile/_iter, with flash_attn_ext_turbo_load_tile
// selected for non-f16 KV tiles). This header adds the missing top half:
//   - a shared-memory size helper that reflects the forced-synchronous (nstages == 0) pipeline
//     the kernel runs for non-f16 KV (cp.async cannot dequantize in flight),
//   - the per-(DKQ, DV, ncols1, ncols2, type_K, type_V) launcher, which hands raw (non-converted)
//     K/V to launch_fattn(need_f16_K/V = false) so the kernel decodes turbo blocks in place,
//   - the dispatch ladder, mirroring ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1/2 in
//     fattn.cu but restricted to tiles Volta can run (ncols1*ncols2 >= 32, no sparse variant).
//
// Instance files: template-instances/fattn-mma-<type_K>-<type_V>-instance-dkq<D>-ncols1_<A>-ncols2_<B>.cu
// Which of them are compiled follows the vec instances (GGML_CUDA_FA_QUANTS -> FA_COMBINATIONS),
// see ggml_cuda_fattn_mma_turbo_instances() in ggml/cmake/common.cmake. The dispatch in fattn.cu
// guards every reference to these cases with the same GGML_CUDA_FA_<K>_<V> macros that gate the
// vec instances, so an uncompiled pair silently keeps the regular kernel selection.

#pragma once

#include "common.cuh"
#include "fattn-mma-f16.cuh"

// Dynamic shared memory the MMA FA kernel requests for a turbo (DKQ, DV, ncols1, ncols2)
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

// Launcher for one (DKQ, DV, ncols1, ncols2, type_K, type_V) TurboQuant MMA case. Explicitly
// instantiated by the fattn-mma-<k>-<v>-instance-*.cu files; fattn.cu only references these
// instantiations (via ggml_cuda_fattn_mma_turbo_case_typed below) when the pair's
// GGML_CUDA_FA_<K>_<V> macro says the instance was compiled, so there is never a dangling
// reference for pairs that GGML_CUDA_FA_QUANTS left out.
template<int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static_assert(type_K != GGML_TYPE_F16 && type_V != GGML_TYPE_F16,
        "turbo MMA cases are for TurboQuant KV types; use ggml_cuda_flash_attn_ext_mma_f16_case for f16 KV");

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa_layout(DKQ, DV, ncols1, ncols2, cc);
    const int warp_size_host = ggml_cuda_info().devices[id].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    static_assert(DKQ != 576, "no turbo MMA instances for Deepseek-style V-is-K-view shapes");

    const size_t nbytes_shared_total = ggml_cuda_fattn_mma_turbo_shared_bytes(DKQ, DV, ncols1, ncols2, cc);

    // The dispatch above is supposed to pick a tile that fits this device's opt-in shared-memory
    // limit. Fail loudly here instead of surfacing an opaque cudaErrorInvalidValue from
    // cudaFuncSetAttribute below.
    if (nbytes_shared_total > ggml_cuda_info().devices[id].smpbo) {
        GGML_ABORT("fatal error: flash_attn_ext_f16<turbo %d,%d,%d,%d> needs %zu bytes of dynamic shared memory, "
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
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, false, false, type_K, type_V>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, false, false, type_K, type_V>;
    }

    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes_shared_total));
        shared_memory_limit_raised[id] = true;
    }

    // need_f16_K/V == false: launch_fattn passes the raw TurboQuant K/V blocks through and the
    // kernel decodes them tile-by-tile via flash_attn_ext_turbo_load_tile.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, false, warp_size_host);
}

template<int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, type_K, type_V)                                        \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                                         \
    <DKQ, DV, ncols1, ncols2, GGML_TYPE_##type_K, GGML_TYPE_##type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// Instantiates the case for whichever symmetric turbo pair was compiled into this build.
// The GGML_CUDA_FA_<K>_<V> macros are defined for every pair by ggml_cuda_fattn_vec_instances()
// in ggml/cmake/common.cmake (same values that gate the vec instance compilation).
template <int DKQ, int DV, int ncols1, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_turbo_case_typed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type type_K = dst->src[1]->type;
    switch (type_K) {
#if defined(GGML_CUDA_FA_TURBO8_0_TURBO8_0) && GGML_CUDA_FA_TURBO8_0_TURBO8_0
        case GGML_TYPE_TURBO8_0:
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, ncols1, ncols2, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO8_0>(ctx, dst);
            return;
#endif
#if defined(GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ) && GGML_CUDA_FA_TURBO3_TCQ_TURBO3_TCQ
        case GGML_TYPE_TURBO3_TCQ:
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, ncols1, ncols2, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ>(ctx, dst);
            return;
#endif
#if defined(GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ) && GGML_CUDA_FA_TURBO2_TCQ_TURBO2_TCQ
        case GGML_TYPE_TURBO2_TCQ:
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, ncols1, ncols2, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ>(ctx, dst);
            return;
#endif
#if defined(GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ) && GGML_CUDA_FA_TURBO1_TCQ_TURBO1_TCQ
        case GGML_TYPE_TURBO1_TCQ:
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, ncols1, ncols2, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ>(ctx, dst);
            return;
#endif
        default:
            GGML_ABORT("fatal error: no TurboQuant MMA instance compiled for K/V type %s (add it to GGML_CUDA_FA_QUANTS)",
                       ggml_type_name(type_K));
    }
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
    // limit (sm_70 has only 96 KB). The turbo helper (nstages == 0) sizes the synchronous
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
