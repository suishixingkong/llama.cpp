// TurboQuant MMA FlashAttention instance (BUUN_MERGE_PLAN.md section 1.4a, Stage 1c L1).
// Symmetric turbo1_tcq pair, DKQ=DV=128, tile ncols1=8 x ncols2=4 (ncols=32). Only tiles
// reachable from the Volta dispatch ladder are instantiated; compilation follows
// GGML_CUDA_FA_QUANTS via ggml_cuda_fattn_mma_turbo_instances() in ggml/cmake/common.cmake.

#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(128, 128, 8, 4, TURBO1_TCQ, TURBO1_TCQ);
