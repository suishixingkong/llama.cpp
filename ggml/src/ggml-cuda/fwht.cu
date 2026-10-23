#include "common.cuh"
#include "fwht.cuh"

#include <cuda_bf16.h>

// Tout is what the row is written as: F32 for the standalone transform, f16/bf16 for the fused transform
// + conversion that a cuBLAS GEMM consumes. `has_signs` folds the +-1 sign vector into the load, which is
// where the graph's separate MUL would have applied it; +-1 products are exact, so the moved multiply
// cannot change the result.
template <int N, typename Tout, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, const float * signs, Tout * dst, const int64_t n_rows, const float scale,
                          const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    // The graph applies the sign vector to a row that may be several transforms wide, so a transform row
    // takes the block at (row % n_blk); with the usual n_blk == 1 that is just the row itself.
    const float * signs_row = has_signs ? signs + (int64_t) (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = (Tout) reg[i];
    }
}

// Wide rows at small row counts (decode): one row per block instead of per warp. The warp kernel
// above serialises every stage on one warp, which is fine for the narrow widths but leaves most of
// the block idle as N grows, so from 512 up a row is spread over FWHT_BLOCK_THREADS threads and the
// stages above the warp width go through shared memory. Same butterfly (ggml_cuda_fwht_block_butterfly),
// same stage order, same place for the scale: the result is bit-identical to the warp kernel, which
// is also what the fused FWHT + q8_1 quantizer relies on.
#define FWHT_BLOCK_THREADS 256

template <int N, int NT, typename Tout, bool has_signs>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const float * src, const float * signs, Tout * dst, const int64_t n_rows, const float scale,
                                const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    // see fwht_cuda: the sign block repeats over the wider row the graph's MUL was applied to
    const float * signs_row = has_signs ? signs + (int64_t) (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = src[i * NT + tid] * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    ggml_cuda_fwht_block_butterfly<N, NT>(reg, s, tid, lane);

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = (Tout) reg[i];
    }
}

// Shared launcher: one instantiation set per (Tout, has_signs), one switch on the width.
template <typename Tout, bool has_signs>
static bool fwht_launch(const float * src_d, const float * signs_d, Tout * dst_d, const int64_t rows,
                        const int n, const int n_blk, const int warp_size, cudaStream_t stream) {
    const float scale = 1 / sqrtf((float) n);

    const int rows_per_block = 4;
    const dim3 grid_warp((unsigned) ((rows + rows_per_block - 1) / rows_per_block), 1, 1);
    const dim3 block_warp(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params warp_params = ggml_cuda_kernel_launch_params(grid_warp, block_warp, 0, stream);

    // One block per row from 512 up (see fwht_cuda_block); the narrow widths keep the warp kernel.
    const dim3 grid_row((unsigned) rows, 1, 1);
    const dim3 block_row(FWHT_BLOCK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params row_params = ggml_cuda_kernel_launch_params(grid_row, block_row, 0, stream);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch((fwht_cuda<64, Tout, has_signs>), warp_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        case 128:
            ggml_cuda_kernel_launch((fwht_cuda<128, Tout, has_signs>), warp_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        case 256:
            ggml_cuda_kernel_launch((fwht_cuda<256, Tout, has_signs>), warp_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        case 512:
            ggml_cuda_kernel_launch((fwht_cuda_block<512, FWHT_BLOCK_THREADS, Tout, has_signs>), row_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        case 1024:
            ggml_cuda_kernel_launch((fwht_cuda_block<1024, FWHT_BLOCK_THREADS, Tout, has_signs>), row_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        case 2048:
            ggml_cuda_kernel_launch((fwht_cuda_block<2048, FWHT_BLOCK_THREADS, Tout, has_signs>), row_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        case 4096:
            ggml_cuda_kernel_launch((fwht_cuda_block<4096, FWHT_BLOCK_THREADS, Tout, has_signs>), row_params, src_d, signs_d, dst_d, rows, scale, n_blk);
            return true;
        default:
            return false;
    }
}

template <typename Tout>
static bool fwht_launch_signed(const float * src_d, const float * signs_d, Tout * dst_d, const int64_t rows,
                               const int n, const int n_blk, const int warp_size, cudaStream_t stream) {
    return signs_d ? fwht_launch<Tout, true> (src_d, signs_d, dst_d, rows, n, n_blk, warp_size, stream)
                   : fwht_launch<Tout, false>(src_d, signs_d, dst_d, rows, n, n_blk, warp_size, stream);
}

bool ggml_cuda_fwht_cast_supported(int n) {
    switch (n) {
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_fwht_cast_to(const ggml_cuda_fwht_cast & plan, void * dst, ggml_type dst_type, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int n = (int) plan.n;
    switch (dst_type) {
        case GGML_TYPE_F32:
            return fwht_launch_signed<float>(plan.src, plan.signs, (float *) dst, plan.rows, n, (int) plan.n_blk, warp_size, stream);
        case GGML_TYPE_F16:
            return fwht_launch_signed<half>(plan.src, plan.signs, (half *) dst, plan.rows, n, (int) plan.n_blk, warp_size, stream);
        case GGML_TYPE_BF16:
            return fwht_launch_signed<nv_bfloat16>(plan.src, plan.signs, (nv_bfloat16 *) dst, plan.rows, n, (int) plan.n_blk, warp_size, stream);
        default:
            return false;
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    return fwht_launch<float, false>(src_d, nullptr, dst_d, rows, n, /*n_blk=*/1, warp_size, ctx.stream());
}
