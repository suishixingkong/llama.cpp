/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Forward declarations for GGML_API symbols defined in this file (satisfies
 * -Wmissing-prototypes under upstream CI's -Werror policy). */
GGML_API void turbo_cpu_fwht(float * x, int group_size);
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size);

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
GGML_API int turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed (exact solution; outer level = 2.1519 sigma) */
static const float CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154496f) return 0;
    if (val < -0.092804f) return 1;
    if (val < -0.044243f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.044243f) return 4;
    if (val <  0.092804f) return 5;
    if (val <  0.154496f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, Lloyd-Max for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.212203f) return 0;
    if (val < -0.162947f) return 1;
    if (val < -0.127026f) return 2;
    if (val < -0.097164f) return 3;
    if (val < -0.070671f) return 4;
    if (val < -0.046174f) return 5;
    if (val < -0.022824f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.022824f) return 8;
    if (val <  0.046174f) return 9;
    if (val <  0.070671f) return 10;
    if (val <  0.097164f) return 11;
    if (val <  0.127026f) return 12;
    if (val <  0.162947f) return 13;
    if (val <  0.212203f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

GGML_API void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- CPU inverse WHT (in-place, group_size elements) ----------
 *
 * Forward is  y = D(s2) * N * H * D(s1) * x   (N = 1/sqrt(group_size))
 * H is the unnormalized Hadamard butterfly with H*H = group_size * I, so
 * (N*H) is self-inverse.  s1 and s2 are ±1 diagonals, also self-inverse.
 * The inverse therefore has the same structure with s1 and s2 swapped:
 *     x = D(s1) * N * H * D(s2) * y
 */
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs2 (undoes the s2 that was applied last in the forward pass)
    for (int i = 0; i < group_size; i++) x[i] *= s2[i];

    // butterfly stages (same as forward — self-inverse up to the inv_sqrt scaling below)
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs1
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s1[i];
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward WHT rotation
        turbo_cpu_fwht(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated, d);

#if TURBO4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids, Lloyd-Max for N(0, 1/sqrt(128))) */
        static const float CENTROIDS_4BIT[16] = {
            -0.241529f, -0.182877f, -0.143016f, -0.111036f,
            -0.083292f, -0.058050f, -0.034299f, -0.011349f,
             0.011349f,  0.034299f,  0.058050f,  0.083292f,
             0.111036f,  0.143016f,  0.182877f,  0.241529f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBO4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBO4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        /* no rnorm: the 4-bit block layout has no such field */
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

#if TURBO4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    static const float CENTROIDS_4BIT[16] = {
        -0.241529f, -0.182877f, -0.143016f, -0.111036f,
        -0.083292f, -0.058050f, -0.034299f, -0.011349f,
         0.011349f,  0.034299f,  0.058050f,  0.083292f,
         0.111036f,  0.143016f,  0.182877f,  0.241529f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
        /* No inverse WHT, dequant stays in the rotated domain.
        * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores.
        * The inverse WHT is applied to the attention output via GGML_OP_TURBO_WHT (direction=1) in the graph. 
        */
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ================================================================== */
/* TQ3_1S / TQ4_1S: WHT-rotated weight quantization                  */
/* ================================================================== */

/* Lloyd-Max centroids for N(0,1) — shared with Metal shaders */
static const float TQ3_0_CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

static const float TQ4_0_CENTROIDS[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

/* WHT sign pattern (golden ratio hash, 32-element blocks) — shared by TQ3 and TQ4 */
static const float TQ3_0_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

#define TQ_BLOCK_SIZE 32
#define TQ_INV_SQRT32 0.17677669529663688f  /* 1/sqrt(32) */

/* Forward RHT: sign flips -> WHT butterfly -> normalize */
static void tq3_0_rht_forward(float * buf) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ3_0_SIGNS[i];
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32;
}

/* Inverse RHT: WHT butterfly -> normalize + unsign */
static void tq3_0_rht_inverse(float * buf) {
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32 * TQ3_0_SIGNS[i];
}

/* Nearest centroid for TQ3 (8 centroids) */
static int tq3_0_choose_index(float val) {
    /* Binary search on midpoints of TQ3_0_CENTROIDS */
    if (val < -1.644041f) return 0;
    if (val < -1.015870f) return 1;
    if (val < -0.493925f) return 2;
    if (val < -0.008701f) return 3;
    if (val <  0.477664f) return 4;
    if (val <  1.001363f) return 5;
    if (val <  1.633223f) return 6;
    return 7;
}

/* Nearest centroid for TQ4 (16 centroids) */
static int tq4_0_choose_index(float val) {
    /* Binary search on midpoints of TQ4_0_CENTROIDS */
    if (val < -2.400804f) return 0;
    if (val < -1.843532f) return 1;
    if (val < -1.437139f) return 2;
    if (val < -1.099286f) return 3;
    if (val < -0.799550f) return 4;
    if (val < -0.522404f) return 5;
    if (val < -0.258222f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.258222f) return 8;
    if (val <  0.522404f) return 9;
    if (val <  0.799550f) return 10;
    if (val <  1.099286f) return 11;
    if (val <  1.437139f) return 12;
    if (val <  1.843532f) return 13;
    if (val <  2.400804f) return 14;
    return 15;
}

/* ---------- TQ3_1S quantization ---------- */

void quantize_row_tq3_1s_ref(const float * GGML_RESTRICT x, block_tq3_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ3_0;
        block_tq3_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float c = TQ3_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float c = TQ3_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ3_0 * 3 / 8);

        /* TQ3 packing: 4 groups of 8 indices packed into 3 bytes each */
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float inv = (j < 16) ? inv0 : inv1;
                indices[i] = (uint8_t)tq3_0_choose_index(buf[j] * inv);
            }
            uint8_t * qp = blk->qs + g * 3;
            qp[0] = (indices[0] & 7) | ((indices[1] & 7) << 3) | ((indices[2] & 3) << 6);
            qp[1] = ((indices[2] >> 2) & 1) | ((indices[3] & 7) << 1) | ((indices[4] & 7) << 4) | ((indices[5] & 1) << 7);
            qp[2] = ((indices[5] >> 1) & 3) | ((indices[6] & 7) << 2) | ((indices[7] & 7) << 5);
        }
    }
}

void dequantize_row_tq3_1s(const block_tq3_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        /* Unpack 3-bit indices */
        float buf[32];
        for (int g = 0; g < 4; g++) {
            const uint8_t * qp = x[blk_i].qs + g * 3;
            uint8_t idx[8];
            idx[0] =  qp[0]       & 7;
            idx[1] = (qp[0] >> 3) & 7;
            idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
            idx[3] = (qp[1] >> 1) & 7;
            idx[4] = (qp[1] >> 4) & 7;
            idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
            idx[6] = (qp[2] >> 2) & 7;
            idx[7] = (qp[2] >> 5) & 7;

            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float d = (j < 16) ? d0 : d1;
                buf[j] = TQ3_0_CENTROIDS[idx[i]] * d;
            }
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ3_0, buf, QK_TQ3_0 * sizeof(float));
    }
}

size_t quantize_tq3_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ3_0 == 0);

    size_t row_size = (n_per_row / QK_TQ3_0) * sizeof(block_tq3_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3_1s_ref(
            src + row * n_per_row,
            (block_tq3_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TQ4_1S quantization ---------- */

void quantize_row_tq4_1s_ref(const float * GGML_RESTRICT x, block_tq4_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ4_1S;
        block_tq4_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float c = TQ4_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float c = TQ4_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack (nibble packing) */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ4_1S / 2);

        for (int j = 0; j < QK_TQ4_1S; j++) {
            float inv = (j < 16) ? inv0 : inv1;
            int idx = tq4_0_choose_index(buf[j] * inv);
            blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j & 1) * 4));
        }
    }
}

void dequantize_row_tq4_1s(const block_tq4_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        float buf[32];
        for (int j = 0; j < 32; j++) {
            uint8_t idx = (x[blk_i].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
            float d = (j < 16) ? d0 : d1;
            buf[j] = TQ4_0_CENTROIDS[idx] * d;
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ4_1S, buf, QK_TQ4_1S * sizeof(float));
    }
}

size_t quantize_tq4_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ4_1S == 0);

    size_t row_size = (n_per_row / QK_TQ4_1S) * sizeof(block_tq4_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq4_1s_ref(
            src + row * n_per_row,
            (block_tq4_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* =====================================================================
 * TurboQuant 8-bit / TCQ KV cache types (ported from buun-llama-cpp)
 *
 * Conventions (IMPORTANT):
 *  - dequantize_* below stay in the ROTATED domain (no inverse FWHT), matching
 *    this tree's turbo2_0/3_0/4_0 convention: the graph applies the forward WHT
 *    to Q (GGML_OP_TURBO_WHT) and the inverse WHT to the attention output.
 *    buun's own CPU refs apply an inverse rotation in dequant instead; that is
 *    NOT this tree's convention and is deliberately not followed here.
 *  - The turbo*_tcq CPU refs are stubs, exactly as in buun: the real encode is a
 *    Viterbi search that only exists on the CUDA/Metal path, and turbo-typed KV
 *    is rejected on backends without TurboQuant support (llama-kv-cache guard).
 * ===================================================================== */

/* ---------- TURBO8_0: 8-bit uniform grid (buun's production format) ---------- */
/*
 * NOTE: buun's *CPU reference* for turbo8 uses a 256-entry Lloyd-Max codebook, but its
 * *CUDA production path* (set_rows / get_rows) uses a closed-form UNIFORM grid
 * centroid[i] = (i - 127.5) / 127.5 with a per-block absmax scale folded into `norm`:
 *
 *   encode: norm = L2(block);  x = block/norm;  x = FWHT(x);
 *           scale = max|x|;    idx = round(x/scale * 127.5 + 127.5)   (clamped to [0,255])
 *           blk.norm = fp16(norm * scale)
 *   decode: y = ((idx - 127.5) / 127.5) * norm
 *
 * The two disagree in buun (its CPU ref is stale), so we follow the CUDA/production
 * format here to keep the CPU and GPU paths compatible.
 */
#define TURBO8_CENTROID(idx) (((float)(idx) - 127.5f) / 127.5f)

void quantize_row_turbo8_0_ref(const float * GGML_RESTRICT x, block_turbo8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO8 == 0);

    const int nb = k / QK_TURBO8;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_TURBO8;

        float norm_sq = 0.0f;
        float buf[QK_TURBO8];
        for (int j = 0; j < QK_TURBO8; j++) {
            buf[j] = src[j];
            norm_sq += buf[j] * buf[j];
        }
        const float norm = sqrtf(norm_sq);
        const float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;

        for (int j = 0; j < QK_TURBO8; j++) buf[j] *= inv_norm;

        /* Forward WHT rotation (matches the CUDA set_rows path). */
        turbo_cpu_fwht(buf, QK_TURBO8);

        /* Per-block absmax scale in the rotated domain. */
        float absmax = 0.0f;
        for (int j = 0; j < QK_TURBO8; j++) absmax = fmaxf(absmax, fabsf(buf[j]));
        const float scale = (absmax > 1e-10f) ? absmax : 1e-10f;
        const float inv_scale = 1.0f / scale;

        for (int j = 0; j < QK_TURBO8; j++) {
            int idx = (int) lrintf(buf[j] * inv_scale * 127.5f + 127.5f);
            idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
            y[block].qs[j] = (uint8_t) idx;
        }

        y[block].norm = GGML_FP32_TO_FP16(norm * scale);
    }
}

void dequantize_row_turbo8_0(const block_turbo8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO8 == 0);
    const int nb = k / QK_TURBO8;
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        /* No inverse WHT: dequant stays in the rotated domain (Q is rotated by the graph). */
        for (int i = 0; i < QK_TURBO8; i++) {
            y[block * QK_TURBO8 + i] = TURBO8_CENTROID(x[block].qs[i]) * norm;
        }
    }
}

size_t quantize_turbo8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO8 == 0);

    const size_t row_size = (n_per_row / QK_TURBO8) * sizeof(block_turbo8_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo8_0_ref(
            src + row * n_per_row,
            (block_turbo8_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_TCQ / TURBO2_TCQ: CPU reference codec ---------- */
/* Trellis-Coded Quantization on a right-shift bitshift trellis with a free initial state.
 *
 * Per 128-element rotation group (the block IS the rotation group, as for turbo8):
 *   1. grp_norm = L2(x); x /= grp_norm;  fwht(x)          -- same FWHT as turbo2/3/4
 *   2. Viterbi over nstates states emitting k bits/step:
 *        cost_new[s] = min{ cost[prev] : prev in the 2^k predecessors of s }
 *                      + (x[t] - cb[s])^2
 *      A state is the sliding window of the last W emitted symbols (W*k bits), the
 *      recurrence is ns = (prev >> k) | (sym << k*(W-1)), so the best predecessor
 *      depends only on s's low k*(W-1) bits: one 2^k-way scan per state group.
 *   3. Backtrack from the cheapest final state; the state left over after the last
 *      backstep is the state *before* step 0.
 *   4. corrected_norm = grp_norm / ||cb[state_t]||_2  (reconstruction is renormalized)
 *   5. Bitstream = (W-1)*k init bits (the pre-step-0 state's low bits) followed by
 *      128 k-bit symbols, LSB-first, zero-padded to the block's byte count.
 *
 * Decode is an O(1) sliding-window lookup: value_t = cb[bits[t*k .. t*k + W*k - 1]] * norm.
 * Like turbo2/3/4, the decode deliberately stays in the WHT-rotated domain (the graph
 * rotates Q as well), so no inverse FWHT is applied.
 *
 * This is the reference the CUDA kernels must reproduce bit-for-bit. The trellis cost
 * uses strictly-less comparisons, i.e. ties resolve to the lowest predecessor and the
 * lowest final state; the CUDA kernels' shuffle reductions resolve ties the same way.
 *
 * Known deviations from buun (see MERGE_BUUN_PLAN.md): buun ships separately trained K/V
 * codebooks (numerically within ~2% of each other) and a decode-side V gain
 * (TURBO_TCQ_ALPHA_V_* = 1.02/1.06 for t3/t2). Neither is taken: this codec is exactly
 * self-consistent (encode norm == decode norm), and the tuning needs a GPU to validate. */

/* TCQ trellis codebooks -- single source of truth, shared with the CUDA kernels. */
#define TURBO_TCQ_TABLE static const
#include "ggml-turbo-tcq-tables.h"

typedef struct {
    int           k;        /* bits per trellis symbol (3 / 2 / 1) */
    int           nstates;  /* 2^(k*w) */
    int           w;        /* symbols in the reconstruction window (3 / 4 / 8) */
    int           nbytes;   /* trellis bitstream payload bytes */
    const float * cb;
} turbo_tcq_geom;

static const turbo_tcq_geom TURBO_TCQ3_GEOM = { 3, 512, 3, 49, TURBO_TCQ3_CODEBOOK };
static const turbo_tcq_geom TURBO_TCQ2_GEOM = { 2, 256, 4, 33, TURBO_TCQ2_CODEBOOK };
static const turbo_tcq_geom TURBO_TCQ1_GEOM = { 1, 256, 8, 17, TURBO_TCQ1_CODEBOOK };

/* Decode one 128-element group. Mirrors the CUDA per-element trellis lookup. */
static void turbo_tcq_decode_group(const uint8_t * GGML_RESTRICT qs, const turbo_tcq_geom * g,
                                   float norm, float * GGML_RESTRICT out) {
    const int k = g->k;
    const int mask = (1 << (k * g->w)) - 1;

    for (int t = 0; t < 128; t++) {
        const int bit_pos  = t * k;
        const int byte_idx = bit_pos >> 3;
        const int bit_off  = bit_pos & 7;
        const uint16_t raw = (uint16_t) qs[byte_idx] | ((uint16_t) qs[byte_idx + 1] << 8);
        out[t] = g->cb[(raw >> bit_off) & mask] * norm;
    }
}

/* Encode one 128-element group: Viterbi + renormalize + bitpack. */
static void turbo_tcq_encode_group(const float * GGML_RESTRICT x, const turbo_tcq_geom * g,
                                   uint8_t * GGML_RESTRICT qs, float * norm_out) {
    const int k        = g->k;
    const int nstates  = g->nstates;
    const int w        = g->w;
    const int npre     = nstates >> k;   /* predecessor groups */
    const int mask     = npre - 1;
    const int shift    = k * (w - 1);    /* == prefix_bits */
    const int sym_mask = (1 << k) - 1;

    float buf[128];
    double norm_sq = 0.0;
    for (int i = 0; i < 128; i++) {
        buf[i]  = x[i];
        norm_sq += (double) buf[i] * (double) buf[i];
    }
    const float grp_norm = (float) sqrt(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    for (int i = 0; i < 128; i++) {
        buf[i] *= inv_norm;
    }

    turbo_cpu_fwht(buf, 128);

    /* Viterbi forward pass, double-buffered. t even: read cost, write cost_b; odd: swap.
     * After 128 steps the final costs live in cost[]. */
    float   cost[512];          /* >= nstates (512 for k=3) */
    float   cost_b[512];
    float   pred_min[128];      /* >= npre = nstates >> k (128 for k=1) */
    uint8_t bt[128 * 128];      /* 128 steps x npre backpointers */

    for (int s = 0; s < nstates; s++) {
        cost[s] = 0.0f;   /* free initial state */
    }

    for (int t = 0; t < 128; t++) {
        const float * cr = (t & 1) ? cost_b : cost;
        float *       cw = (t & 1) ? cost   : cost_b;
        const float   xt = buf[t];

        for (int pid = 0; pid < npre; pid++) {
            const int base = pid << k;
            float best   = cr[base];
            int   best_p = 0;
            for (int p = 1; p < (1 << k); p++) {
                const float c = cr[base | p];
                if (c < best) {
                    best   = c;
                    best_p = p;
                }
            }
            bt[t * npre + pid] = (uint8_t) best_p;
            pred_min[pid]      = best;
        }

        for (int s = 0; s < nstates; s++) {
            const float d = xt - g->cb[s];
            cw[s] = pred_min[s & mask] + d * d;
        }
    }

    /* Cheapest final state; ties resolve to the lowest state index. */
    int best_state = 0;
    for (int s = 1; s < nstates; s++) {
        if (cost[s] < cost[best_state]) {
            best_state = s;
        }
    }

    /* Backtrack. outputs[t] is the symbol emitted at step t; the state left after the
     * final backstep is the state before step 0 (call it the initial state). */
    uint8_t outputs[128];
    int state = best_state;
    for (int t = 127; t >= 0; t--) {
        outputs[t] = (uint8_t) (state >> shift);
        const int p = bt[t * npre + (state & mask)];
        state = ((state & mask) << k) | p;
    }
    const int initial_state = state;

    /* Reconstruction norm: ||cb[state_t]||_2 over the group. For t < w-1 the state is
     * replayed from the initial state (it includes the virtual pre-group symbols). */
    double recon_sq = 0.0;
    for (int t = 0; t < 128; t++) {
        int cur;
        if (t < w - 1) {
            cur = initial_state;
            for (int u = 0; u <= t; u++) {
                cur = (cur >> k) | ((int) outputs[u] << shift);
            }
        } else {
            cur = 0;
            for (int b = 0; b < w; b++) {
                cur |= ((int) outputs[t - (w - 1) + b] & sym_mask) << (b * k);
            }
        }
        const float c = g->cb[cur];
        recon_sq += (double) c * (double) c;
    }
    const float recon_norm = (float) sqrt(recon_sq);
    *norm_out = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    /* Bitpack: shift bits (the pre-step-0 state's low bits), then the 128 symbols. */
    const int init_bits  = (initial_state >> k) & ((1 << shift) - 1);
    const int total_bits = shift + 128 * k;

    memset(qs, 0, (size_t) g->nbytes);
    for (int pos = 0; pos < total_bits; pos++) {
        int v;
        if (pos < shift) {
            v = (init_bits >> pos) & 1;
        } else {
            const int sb = pos - shift;
            v = (outputs[sb / k] >> (sb % k)) & 1;
        }
        qs[pos >> 3] |= (uint8_t) (v << (pos & 7));
    }
}

void quantize_row_turbo3_tcq_ref(const float * GGML_RESTRICT x, block_turbo3_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_TCQ == 0);
    const int64_t nb = k / QK_TURBO3_TCQ;
    for (int64_t b = 0; b < nb; b++) {
        float norm;
        turbo_tcq_encode_group(x + b * QK_TURBO3_TCQ, &TURBO_TCQ3_GEOM, y[b].qs, &norm);
        y[b].norm = GGML_FP32_TO_FP16(norm);
        y[b].pad  = 0;
    }
}

void dequantize_row_turbo3_tcq(const block_turbo3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_TCQ == 0);
    const int64_t nb = k / QK_TURBO3_TCQ;
    for (int64_t b = 0; b < nb; b++) {
        turbo_tcq_decode_group(x[b].qs, &TURBO_TCQ3_GEOM, GGML_FP16_TO_FP32(x[b].norm),
                               y + b * QK_TURBO3_TCQ);
    }
}

size_t quantize_turbo3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3_TCQ == 0);

    const size_t row_size = (n_per_row / QK_TURBO3_TCQ) * sizeof(block_turbo3_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_tcq_ref(
            src + row * n_per_row,
            (block_turbo3_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

void quantize_row_turbo2_tcq_ref(const float * GGML_RESTRICT x, block_turbo2_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2_TCQ == 0);
    const int64_t nb = k / QK_TURBO2_TCQ;
    for (int64_t b = 0; b < nb; b++) {
        float norm;
        turbo_tcq_encode_group(x + b * QK_TURBO2_TCQ, &TURBO_TCQ2_GEOM, y[b].qs, &norm);
        y[b].norm = GGML_FP32_TO_FP16(norm);
        y[b].pad  = 0;
    }
}

void dequantize_row_turbo2_tcq(const block_turbo2_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2_TCQ == 0);
    const int64_t nb = k / QK_TURBO2_TCQ;
    for (int64_t b = 0; b < nb; b++) {
        turbo_tcq_decode_group(x[b].qs, &TURBO_TCQ2_GEOM, GGML_FP16_TO_FP32(x[b].norm),
                               y + b * QK_TURBO2_TCQ);
    }
}

size_t quantize_turbo2_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2_TCQ == 0);

    const size_t row_size = (n_per_row / QK_TURBO2_TCQ) * sizeof(block_turbo2_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_tcq_ref(
            src + row * n_per_row,
            (block_turbo2_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO1_TCQ (1.25 bpv) ---------- */
/* Same right-shift trellis, k=1, w=8, 256 states, 7-bit init prefix, so the generic core
 * above handles it unchanged.
 *
 * Convention resolved from the graph, not from the fork. buun's FA path carries TWO
 * decoders for this tier: one applying a per-row inverse FWHT (emitting the original
 * domain) and one emitting the stored rotated coefficients. Which one is correct is not a
 * property of the codec -- the stored bits are identical either way -- it follows from how
 * the graph compensates the rotation:
 *
 *   this tree (llama-graph.cpp): Q is rotated FORWARD and K stays rotated in the cache, so
 *   the scores need no un-rotation; V stays rotated too and the graph applies the INVERSE
 *   WHT to the attention output (both the FA output and the non-FA `kqv`). Both gates
 *   already list GGML_TYPE_TURBO1_TCQ.
 *
 * So this tier decodes in the ROTATED domain here, exactly like turbo2/3_tcq and
 * turbo2/3/4/8 -- i.e. buun's `..._dequant_f16_rot` variant, whose own comment says it
 * mirrors the turbo2/3_tcq contract. buun's inverse-FWHT-at-dequant variant belongs to a
 * different graph configuration and is deliberately not taken. */

void quantize_row_turbo1_tcq_ref(const float * GGML_RESTRICT x, block_turbo1_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO1_TCQ == 0);
    const int64_t nb = k / QK_TURBO1_TCQ;
    for (int64_t b = 0; b < nb; b++) {
        float norm;
        turbo_tcq_encode_group(x + b * QK_TURBO1_TCQ, &TURBO_TCQ1_GEOM, y[b].qs, &norm);
        y[b].norm = GGML_FP32_TO_FP16(norm);
        y[b].pad  = 0;
    }
}

void dequantize_row_turbo1_tcq(const block_turbo1_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO1_TCQ == 0);
    const int64_t nb = k / QK_TURBO1_TCQ;
    for (int64_t b = 0; b < nb; b++) {
        turbo_tcq_decode_group(x[b].qs, &TURBO_TCQ1_GEOM, GGML_FP16_TO_FP32(x[b].norm),
                               y + b * QK_TURBO1_TCQ);
    }
}

size_t quantize_turbo1_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO1_TCQ == 0);

    const size_t row_size = (n_per_row / QK_TURBO1_TCQ) * sizeof(block_turbo1_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo1_tcq_ref(
            src + row * n_per_row,
            (block_turbo1_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
