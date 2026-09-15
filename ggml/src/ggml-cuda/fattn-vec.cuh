#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
// ncols1 == number of Q *tokens* per block (batch dimension).
// ncols2 == number of Q *heads* sharing one KV head that are packed into a single block.
//     ncols2 > 1 makes each block load a KV tile once and reuse it across ncols2 query heads,
//     which is where the KV re-read redundancy of grouped-query attention is removed.
//     Requires gqa_ratio % ncols2 == 0 and ne02 % ncols2 == 0, enforced by the caller.
template<int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 2)
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int ncols = ncols1*ncols2;

    // Q_reg and VKQ are per-thread arrays that are replicated for every one of the ncols
    // columns, so packing ncols2 heads into a block multiplies them by ncols2. Spreading
    // each D-row over more threads shrinks them by the same factor and keeps them in
    // registers; D/(2*cpy_ne) is the widest spread the 16 B copy granularity allows.
    // This is exactly the thread distribution the quantized-K/V path already uses.
    constexpr int nthreads_packed = D/(2*cpy_ne) < WARP_SIZE ? D/(2*cpy_ne) : WARP_SIZE;
    constexpr int nthreads_f      = ncols2 > 1 ? nthreads_packed : 128 / cpy_nb;

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    // Turbo3 uses the float Q path (like f16/bf16), not q8_1 integer path
    constexpr bool K_is_unquantized = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 || type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO4_0);
    constexpr bool V_is_unquantized = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16 || type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0);
    constexpr bool K_is_turbo = (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO4_0);
    // Turbo KQ dot does byte extraction + centroid lookup + scalar mul, not vectorized f16 loads.
    // nthreads_KQ=1: each thread computes a full KQ product alone — eliminates warp_reduce_sum
    // shuffle and halves KQ loop iterations. Each thread holds full Q vector in registers.
    constexpr int nthreads_KQ = K_is_turbo ? 1 : (K_is_unquantized ? nthreads_f : nthreads_KQ_q);
    constexpr bool V_is_turbo = (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0);
    // Turbo V dequant is scalar (byte extract + LUT), not vectorized loads.
    // Halve nthreads_V to double V_cols_per_iter (process 2 V rows per loop iteration),
    // reducing loop overhead and improving ILP in the V aggregation phase.
    // Eighth nthreads_V for turbo: V_cols_per_iter goes from 4→8, processing 8 V positions
    // per outer loop iteration. Halves outer loop count again, more ILP from concurrent V rows.
    constexpr int nthreads_V  = V_is_unquantized ? (V_is_turbo ? (nthreads_V_q / 8 < 1 ? 1 : nthreads_V_q / 8) : nthreads_f) : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = V_is_unquantized ? ((type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0) ? 4 : 2*cpy_ne) : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = !K_is_unquantized;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols1; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0 = blockIdx.z*ncols2 - sequence*ne02; // == (blockIdx.z % (ne02/ncols2)) * ncols2
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    // All ncols2 packed heads share one KV head because gqa_ratio % ncols2 == 0.
    Q += nb03*sequence + nb02* head0              + nb01*ic0;
    K += nb13*sequence + nb12*(head0 / gqa_ratio);
    V += nb23*sequence + nb22*(head0 / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    float slope[ncols2];
#pragma unroll
    for (int c = 0; c < ncols2; ++c) {
        slope[c] = get_alibi_slope(max_bias, head0 + c, n_head_log2, m0, m1);
    }

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    // Shared-memory LUT for turbo KQ scoring: precompute Q[d] * centroid[c] once,
    // then the hot loop does turbo_lut[d][idx] (shmem read, no multiply).
    // turbo4 excluded: 16 centroids × D exceeds shmem budget.
    // Stride = n_centroids+1 to avoid bank conflicts.
    constexpr int n_centroids_lut = (D <= 256 && type_K == GGML_TYPE_TURBO3_0) ? 8 :
                                    (D <= 256 && type_K == GGML_TYPE_TURBO2_0) ? 4 : 0;
    constexpr int lut_stride = n_centroids_lut > 0 ? n_centroids_lut + 1 : 1;
    __shared__ half turbo_lut[n_centroids_lut > 0 ? D : 1][lut_stride];

    // Sparse V: skip V dequant for positions with negligible attention weights.
    // At long context, most V positions contribute < 1e-6 to the output — skipping
    // their dequant saves significant compute (especially for quantized V types).
    constexpr float sparse_v_threshold_f = 1e-6f;
#ifdef V_DOT2_F32_F16_AVAILABLE
    const     half  sparse_v_threshold_h = __float2half(sparse_v_threshold_f);
#endif

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols1 > 1 && ic0 + j/ncols2 >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + (j/ncols2)*nb01 + (j%ncols2)*nb02);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + (j/ncols2)*nb01 + (j%ncols2)*nb02);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols1 == 1 || ic0 + j/ncols2 < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + (j/ncols2)*nb01 + (j%ncols2)*nb02);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols1 == 1 || ic0 + j/ncols2 < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    // Build shared-memory LUT: turbo_lut[d][c] = half(Q[d] * scale * centroid[c])
    if constexpr (n_centroids_lut > 0 && ncols == 1) {
        const float * centroids_ptr = (type_K == GGML_TYPE_TURBO3_0) ? TURBO_CENTROIDS_3BIT :
                                      TURBO_CENTROIDS_2BIT;
        const float * Q_f = (const float *)(Q + 0*nb01);
        for (int d = tid; d < D; d += nthreads) {
            const float q_val = Q_f[d] * scale;
            for (int c = 0; c < n_centroids_lut; c++) {
                turbo_lut[d][c] = __float2half(q_val * centroids_ptr[c]);
            }
        }
        __syncthreads();
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (n_centroids_lut > 0 && ncols == 1 && type_K == GGML_TYPE_TURBO3_0) {
                    // LUT scoring: 8 elements per iteration (2 qs bytes + 1 signs byte)
                    const block_turbo3_0 * K_turbo = (const block_turbo3_0 *)(K + i_KQ*nb11);
                    sum = 0.0f;
                    for (int d0 = 0; d0 < D; d0 += 8) {
                        const int ib = d0 / QK_TURBO3;
                        const int jj = d0 % QK_TURBO3;
                        const float norm = __half2float(K_turbo[ib].norm);
                        const uint8_t qs0 = K_turbo[ib].qs[jj / 4];
                        const uint8_t qs1 = K_turbo[ib].qs[jj / 4 + 1];
                        const uint8_t sgn = K_turbo[ib].signs[jj / 8];
                        sum += (__half2float(turbo_lut[d0  ][((qs0>>0)&3)|((sgn>>0&1)<<2)]) +
                                __half2float(turbo_lut[d0+1][((qs0>>2)&3)|((sgn>>1&1)<<2)]) +
                                __half2float(turbo_lut[d0+2][((qs0>>4)&3)|((sgn>>2&1)<<2)]) +
                                __half2float(turbo_lut[d0+3][((qs0>>6)&3)|((sgn>>3&1)<<2)]) +
                                __half2float(turbo_lut[d0+4][((qs1>>0)&3)|((sgn>>4&1)<<2)]) +
                                __half2float(turbo_lut[d0+5][((qs1>>2)&3)|((sgn>>5&1)<<2)]) +
                                __half2float(turbo_lut[d0+6][((qs1>>4)&3)|((sgn>>6&1)<<2)]) +
                                __half2float(turbo_lut[d0+7][((qs1>>6)&3)|((sgn>>7&1)<<2)])) * norm;
                    }
                } else if constexpr (n_centroids_lut > 0 && ncols == 1 && type_K == GGML_TYPE_TURBO2_0) {
                    // LUT scoring for turbo2: 8 elements per iteration (2 qs bytes, no signs)
                    const block_turbo2_0 * K_turbo = (const block_turbo2_0 *)(K + i_KQ*nb11);
                    sum = 0.0f;
                    for (int d0 = 0; d0 < D; d0 += 8) {
                        const int ib = d0 / QK_TURBO2;
                        const int jj = d0 % QK_TURBO2;
                        const float norm = __half2float(K_turbo[ib].norm);
                        const uint8_t qs0 = K_turbo[ib].qs[jj / 4];
                        const uint8_t qs1 = K_turbo[ib].qs[jj / 4 + 1];
                        sum += (__half2float(turbo_lut[d0  ][(qs0>>0)&3]) +
                                __half2float(turbo_lut[d0+1][(qs0>>2)&3]) +
                                __half2float(turbo_lut[d0+2][(qs0>>4)&3]) +
                                __half2float(turbo_lut[d0+3][(qs0>>6)&3]) +
                                __half2float(turbo_lut[d0+4][(qs1>>0)&3]) +
                                __half2float(turbo_lut[d0+5][(qs1>>2)&3]) +
                                __half2float(turbo_lut[d0+6][(qs1>>4)&3]) +
                                __half2float(turbo_lut[d0+7][(qs1>>6)&3])) * norm;
                    }
                } else {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                }

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols1 == 1 || ic0 + j/ncols2 < int(ne01.z))) {
                    sum += slope[j % ncols2]*__half2float(maskh[(j/ncols2)*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = __expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = __expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            if constexpr (!V_is_turbo) { KQ[j*nthreads + tid] = KQ_reg[j]; }

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

        ggml_cuda_syncwarp();

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (V_is_turbo) {
                    const float kq_val = __shfl_sync(0xFFFFFFFF, KQ_reg[j], k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V));
                    KQ_k[j] = make_half2(__float2half(kq_val), __float2half(kq_val));
                } else {
                    KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
                }
            }

            // Sparse V: skip V dequant if all attention weights for this position are negligible.
            // For turbo types, the check is compiled out: at typical decode context lengths
            // (< ~4K tokens) with threshold 1e-6, no positions are ever skipped, so the
            // per-position branch is pure overhead (misprediction + comparison cost). This
            // also dodges the warp-divergence regression on turbo paths that motivated the
            // April 24 revert (commit f2dc968).
            if constexpr (!V_is_turbo) {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (__hgt(__low2half(KQ_k[j]), sparse_v_threshold_h)) { dominated = false; break; }
                }
                if (dominated) { continue; }
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (V_is_turbo) {
                    KQ_k[j] = __shfl_sync(0xFFFFFFFF, KQ_reg[j], k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V));
                } else {
                    KQ_k[j] = KQ[j*nthreads + k];
                }
            }

            // Sparse V: skip V dequant if all attention weights for this position are negligible.
            // Compiled out for turbo types — see half2 path comment above.
            if constexpr (!V_is_turbo) {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (KQ_k[j] >= sparse_v_threshold_f) { dominated = false; break; }
                }
                if (dominated) { continue; }
            }

            // Turbo V path: precompute scaled centroids once per block to eliminate
            // per-element norm multiply.  centroid[idx]*norm is computed 8/4/16 times
            // (once per centroid) instead of D times (once per element).
            if constexpr (type_V == GGML_TYPE_TURBO3_0) {
                const block_turbo3_0 * vb = (const block_turbo3_0 *)(V + k*nb21);
                int prev_ib = -1;
                float sc[8];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO3;
                    const int j0 = i0 % QK_TURBO3;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
#pragma unroll
                        for (int c = 0; c < 8; ++c) { sc[c] = TURBO_CENTROIDS_3BIT[c] * norm; }
                    }

                    const uint8_t qs_byte  = vb[ib].qs[j0 / 4];
                    const uint8_t sgn_byte = vb[ib].signs[j0 / 8];
                    const int     shift_s  = j0 % 8;

                    const uint8_t idx0 = ((qs_byte >> 0) & 0x3) | (((sgn_byte >> (shift_s+0)) & 0x1) << 2);
                    const uint8_t idx1 = ((qs_byte >> 2) & 0x3) | (((sgn_byte >> (shift_s+1)) & 0x1) << 2);
                    const uint8_t idx2 = ((qs_byte >> 4) & 0x3) | (((sgn_byte >> (shift_s+2)) & 0x1) << 2);
                    const uint8_t idx3 = ((qs_byte >> 6) & 0x3) | (((sgn_byte >> (shift_s+3)) & 0x1) << 2);

#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else if constexpr (type_V == GGML_TYPE_TURBO2_0) {
                const block_turbo2_0 * vb = (const block_turbo2_0 *)(V + k*nb21);
                int prev_ib = -1;
                float sc[4];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO2;
                    const int j0 = i0 % QK_TURBO2;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
#pragma unroll
                        for (int c = 0; c < 4; ++c) { sc[c] = TURBO_CENTROIDS_2BIT[c] * norm; }
                    }

                    const uint8_t qs_byte = vb[ib].qs[j0 / 4];

                    const uint8_t idx0 = (qs_byte >> 0) & 0x3;
                    const uint8_t idx1 = (qs_byte >> 2) & 0x3;
                    const uint8_t idx2 = (qs_byte >> 4) & 0x3;
                    const uint8_t idx3 = (qs_byte >> 6) & 0x3;

#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {
                const block_turbo4_0 * vb = (const block_turbo4_0 *)(V + k*nb21);
                int prev_ib = -1;
                float sc[16];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO4;
                    const int j0 = i0 % QK_TURBO4;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
#pragma unroll
                        for (int c = 0; c < 16; ++c) { sc[c] = TURBO_CENTROIDS_4BIT[c] * norm; }
                    }

                    const uint8_t qs_byte0 = vb[ib].qs[j0 / 2];
                    const uint8_t qs_byte1 = vb[ib].qs[j0 / 2 + 1];

                    const uint8_t idx0 = (qs_byte0 >> 0) & 0xF;
                    const uint8_t idx1 = (qs_byte0 >> 4) & 0xF;
                    const uint8_t idx2 = (qs_byte1 >> 0) & 0xF;
                    const uint8_t idx3 = (qs_byte1 >> 4) & 0xF;

#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else {
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                        }
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float sink = ((const float *) sinks)[head0 + j % ncols2];

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = __expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? __expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols1 > 1 && ic0 + j_VKQ/ncols2 >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = __expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ/ncols2)*ne02 + head0 + j_VKQ%ncols2)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    // Unrolled so that KQ_max/KQ_sum are indexed by a compile-time constant; a runtime
    // index would force those register arrays into local memory.
    if (gridDim.y != 1 && tid < ncols) {
#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            if (tid != jc || (ncols1 > 1 && ic0 + jc/ncols2 >= int(ne01.z))) {
                continue;
            }
            dst_meta[((sequence*int(ne01.z) + ic0 + jc/ncols2)*ne02 + head0 + jc%ncols2)*gridDim.y + blockIdx.y] =
                make_float2(KQ_max[jc], KQ_sum[jc]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template<int D, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_partial_case_impl(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float * partial_dst,
        float2 * partial_meta,
        int nparts) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    constexpr int ncols = 1;
    constexpr int nthreads = 128;
    constexpr int nwarps = nthreads/WARP_SIZE;
    fattn_kernel_t kernel = flash_attn_ext_vec<D, ncols, 1, type_K, type_V, use_logit_softcap>;

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias       )/n_head_log2);
    const float m1 = powf(2.0f, -(max_bias/2.0f)/n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    const dim3 blocks(Q->ne[1], nparts, Q->ne[2]*Q->ne[3]);
    const dim3 threads(WARP_SIZE, nwarps, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        nullptr,
        nullptr,
        partial_dst,
        partial_meta,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());
}

template<int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_partial_case(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float * partial_dst,
        float2 * partial_meta,
        int nparts) {
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap == 0.0f) {
        ggml_cuda_flash_attn_ext_vec_partial_case_impl<
            D, type_K, type_V, false>(ctx, dst, partial_dst, partial_meta, nparts);
    } else {
        ggml_cuda_flash_attn_ext_vec_partial_case_impl<
            D, type_K, type_V, true>(ctx, dst, partial_dst, partial_meta, nparts);
    }
}

template <int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, ncols1, ncols2, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        // GQA head packing. The ncols2 ladders in ggml_cuda_get_best_fattn_kernel and in the
        // MMA path only take powers of two, so a gqa_ratio with an odd factor of 3 (e.g. 6)
        // gets no KV reuse at all in this kernel: one block per query head, hence gqa_ratio
        // independent streams over the same KV head. Packing 3 heads per block removes that
        // factor. Powers of two are deliberately not handled here - those shapes are routed
        // to the TILE/MMA kernels by the selector, which already pack them.
        // Restricted to unquantized K/V on purpose. The register relief that makes packing
        // fit (spreading each D-row over D/(2*cpy_ne) threads) only applies to the F16/BF16
        // dot product; the quantized path already uses that distribution and already sits at
        // ~252-255 registers at D == 256, so a packed quantized variant spills its VKQ
        // accumulator inside the main loop (measured: 30 STL + 17 LDL in the hot loop).
        // Quantized KV therefore keeps the existing, unpacked kernel unchanged.
        constexpr bool packing_supported =
            (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) &&
            (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16);

        // if constexpr, so the packed kernel is not even instantiated for quantized K/V.
        if constexpr (packing_supported) {
            const ggml_tensor * K = dst->src[1];
            const int gqa_ratio = Q->ne[2] / K->ne[2];

            if (gqa_ratio % 3 == 0 && Q->ne[2] % 3 == 0) {
                constexpr int ncols1 = 1;
                constexpr int ncols2 = 3;
                if (logit_softcap == 0.0f) {
                    constexpr bool use_logit_softcap = false;
                    ggml_cuda_flash_attn_ext_vec_case_impl<D, ncols1, ncols2, type_K, type_V, use_logit_softcap>(ctx, dst);
                } else {
                    constexpr bool use_logit_softcap = true;
                    ggml_cuda_flash_attn_ext_vec_case_impl<D, ncols1, ncols2, type_K, type_V, use_logit_softcap>(ctx, dst);
                }
                return;
            }
        }

        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, 1, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, 1, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, 1, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, 1, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                                      \
    template void ggml_cuda_flash_attn_ext_vec_case                                 \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);         \
    template void ggml_cuda_flash_attn_ext_vec_partial_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst,          \
        float * partial_dst, float2 * partial_meta, int nparts)

#define EXTERN_DECL_FATTN_VEC_CASE(D, type_K, type_V)                               \
    extern template void ggml_cuda_flash_attn_ext_vec_case                          \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);         \
    extern template void ggml_cuda_flash_attn_ext_vec_partial_case                  \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst,          \
        float * partial_dst, float2 * partial_meta, int nparts)

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)                     \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);          \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0);         \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1);         \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0);         \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1);         \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0);         \
    EXTERN_DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16);

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

// TurboQuant3 — turbo3 K + turbo3 V (KV cache uses same type)
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);

// Mixed turbo3/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);

// Mixed f16/turbo3 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);

// TurboQuant2 -- turbo2 K + turbo2 V
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);

// Mixed turbo2/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);

// Mixed f16/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);

// Mixed turbo3/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);

// TurboQuant4 — turbo4 K + turbo4 V (KV cache uses same type)
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);

// Mixed f16/turbo4 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);

// Mixed turbo4/turbo3 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
