// Non-f16 KV MMA FlashAttention instance (MERGE_BUUN_PLAN.md section 1.4a).
// <K>-<V> pair, DKQ=DV=128, tile ncols1=8 x ncols2=8 (ncols=64). Only tiles
// reachable from the Volta dispatch ladder are instantiated; compilation follows
// GGML_CUDA_FA_QUANTS via ggml_cuda_fattn_mma_turbo_instances() in ggml/cmake/common.cmake.

#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(128, 128, 8, 8, Q5_1, Q4_1);
