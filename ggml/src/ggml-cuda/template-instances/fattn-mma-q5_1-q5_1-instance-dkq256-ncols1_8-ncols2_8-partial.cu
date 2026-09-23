// Non-f16 KV MMA FlashAttention PARTIAL-output instance for the kv-stream span prefill
// (BUUN_MERGE_PLAN.md section 10.10): D=256, tile ncols1=8 x ncols2=8, the only shape the
// span path instantiates. Compiled for the turbo pairs and the mixed/symmetric quantized
// pairs listed below; every other pair keeps the f16-conversion route. Compilation follows
// GGML_CUDA_FA_QUANTS.

#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_PARTIAL_CASE(256, 256, 8, 8, Q5_1, Q5_1);
