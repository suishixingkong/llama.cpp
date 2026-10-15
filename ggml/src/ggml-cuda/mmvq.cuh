#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// fork buun-llama-cpp: MoE expert cache dispatch. Runs one batched matvec over
// slot-pool experts selected by device-side IDs.
// Output row c is W[ids[c]] times activation row act_ids[c].
// slot_stride_bytes is the original tensor nb[2], and act_q8 uses the padded q8_1 layout from quantize_row_q8_1_cuda.
// force_dedicated selects the MoE kernel even when modulo routing needs no explicit activation map.
//
// PORT-DEVIATION: this tree carries the generic modulo-index path only; the
// dedicated activation-map kernel needs the fork's evolved MoE kernel and
// ggml_cuda_mmvq_fusion_args_device. force_dedicated and act_ids_dev are
// therefore accepted and ignored, and the returned path is always generic.
// See docs/MERGE_BUUN_PLAN.md 10.92.3.
enum class ggml_cuda_moe_cache_mmv_path {
    generic,
    dedicated,
};

ggml_cuda_moe_cache_mmv_path ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    bool force_dedicated, cudaStream_t stream);

// Runs up * GLU(gate) for rows whose two independently cached experts are resident.
// PORT-DEVIATION: unreachable in this tree -- moe_cache_fused_begin() returns
// nullptr before it can plan a fused node. Kept as a hard failure so a gate
// that ever regresses is loud instead of silently wrong.
void ggml_cuda_moe_cache_mmv_fused(
    const void * up_pool, const void * gate_pool, ggml_type type0,
    const char * act_q8, const int32_t * up_ids_dev,
    const int32_t * gate_ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    float up_min, float up_max, float gate_min, float gate_max,
    cudaStream_t stream);
