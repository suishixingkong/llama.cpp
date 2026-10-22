#pragma once

#include "common.cuh"

// In-register block FWHT over N floats held as NE = N/NT per thread, element i*NT + tid in reg[i].
// Shared by the standalone kernel (fwht_cuda_block) and the fused FWHT + q8_1 quantizer so both
// produce bit-identical transforms. `s` must hold N floats.
//
// The stages run in ascending h over the same pairs (i, i^h) with the low element taking x + y and
// the high one x - y, exactly like the warp-per-row kernel above it, so the transform value does not
// depend on where the row is staged (registers vs shared memory) or on how many threads hold it:
// only the sequence of additions matters, and that sequence is identical.
template <int N, int NT>
__device__ __forceinline__ void ggml_cuda_fwht_block_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// Widths for which a fused transform + conversion exists (the same set ggml_cuda_op_fwht handles).
bool ggml_cuda_fwht_cast_supported(int n);

// Fused transform + conversion: transforms `plan.src` (optionally multiplied by `plan.signs`) row by row
// and writes the result as `dst_type` elements to `dst`. This is the copy the GEMM's src1 conversion would
// have produced from the F32 rows, so the F32 rows never have to exist. `dst_type` F32 means "write the
// transform in place", which is what the GEMM does when it has no conversion to perform.
bool ggml_cuda_fwht_cast_to(const ggml_cuda_fwht_cast & plan, void * dst, ggml_type dst_type, cudaStream_t stream);
