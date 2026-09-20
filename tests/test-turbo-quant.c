#include <stdio.h>
#include <math.h>
#include <string.h>

#include "ggml.h"

/* TurboQuant CPU round-trip test.
 *
 * turbo3_0 / turbo4_0 / turbo8_0 all keep their dequantized output in the
 * WHT-rotated domain: the graph rotates Q with GGML_OP_TURBO_WHT as well, so
 * <Q_rot, K_rot> already yields correct attention scores without an explicit
 * inverse. To compare against the original input the test therefore applies
 * turbo_cpu_fwht_inverse() afterwards, one 128-element rotation group at a time.
 *
 * The test exits non-zero if any check fails, so it doubles as the regression
 * gate for the TurboQuant port (the CPU encode/decode is the reference format
 * the CUDA kernels have to match).
 */

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void quantize_row_turbo8_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo8_0(const void * x, float * y, long long k);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

#define QK_TURBO8   128
#define MAX_BLOCKS  8

/* Mirrors block_turbo8_0 from ggml-common.h: fp16 norm + 128 one-byte indices. */
typedef struct {
    unsigned short norm;
    unsigned char  qs[QK_TURBO8];
} blk_turbo8_mirror;

static int g_checks = 0;
static int g_fail   = 0;

static int check_ge(const char * name, double got, double min_ok) {
    const int ok = got >= min_ok;
    g_checks++;
    if (!ok) { g_fail++; }
    printf("  [%s] %s: %.8g (>= %.8g)\n", ok ? "PASS" : "FAIL", name, got, min_ok);
    return ok;
}

static int check_le(const char * name, double got, double max_ok) {
    const int ok = got <= max_ok;
    g_checks++;
    if (!ok) { g_fail++; }
    printf("  [%s] %s: %.8g (<= %.8g)\n", ok ? "PASS" : "FAIL", name, got, max_ok);
    return ok;
}

static int check_eq_i(const char * name, long long got, long long want) {
    const int ok = got == want;
    g_checks++;
    if (!ok) { g_fail++; }
    printf("  [%s] %s: %lld (want %lld)\n", ok ? "PASS" : "FAIL", name, got, want);
    return ok;
}

static void stats(const float * in, const float * out, int n,
                  double * mse, double * cos_sim, double * in_norm, double * out_norm) {
    double se = 0.0, dot = 0.0, ni = 0.0, no = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double) in[i] - (double) out[i];
        se  += d * d;
        dot += (double) in[i] * (double) out[i];
        ni  += (double) in[i] * (double) in[i];
        no  += (double) out[i] * (double) out[i];
    }
    *mse      = se / n;
    *cos_sim  = (ni > 0.0 && no > 0.0) ? dot / sqrt(ni) / sqrt(no) : 0.0;
    *in_norm  = sqrt(ni);
    *out_norm = sqrt(no);
}

static void print_vec(const char * tag, const float * v) {
    printf("  %s [%.4f, %.4f, %.4f, %.4f]\n", tag, v[0], v[1], v[2], v[3]);
}

/* Deterministic pseudo-random in [-1, 1). */
static float rnd(void) {
    static unsigned int s = 12345u;
    s = s * 1664525u + 1013904223u;
    return (float) (s >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

/* quantize -> dequantize -> inverse WHT, one 128-element group per block. */
static void roundtrip8(const float * in, float * out, int n, unsigned char * scratch) {
    quantize_row_turbo8_0_ref(in, scratch, n);
    dequantize_row_turbo8_0(scratch, out, n);
    for (int b = 0; b < n / QK_TURBO8; b++) {
        turbo_cpu_fwht_inverse(out + b * QK_TURBO8, QK_TURBO8);
    }
}

int main(void) {
    const int d = 128;
    static union {
        blk_turbo8_mirror b[MAX_BLOCKS];
        unsigned char     bytes[MAX_BLOCKS * sizeof(blk_turbo8_mirror)];
    } store;
    static float input[1024], output[1024];
    double mse, cosv, in_norm, out_norm;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    /* ---------------------------------------------------------------
     * Test 0: registration / layout
     * --------------------------------------------------------------- */
    printf("Test 0 (turbo8): ggml registration & block layout\n");
    check_eq_i("sizeof(block_turbo8_0) mirror", (long long) sizeof(blk_turbo8_mirror), 130);

    const struct ggml_type_traits * tr8 = ggml_get_type_traits(GGML_TYPE_TURBO8_0);
    if (tr8 == NULL) {
        g_checks++; g_fail++;
        printf("  [FAIL] ggml_get_type_traits(GGML_TYPE_TURBO8_0) returned NULL\n");
    } else {
        check_eq_i("traits blck_size",  (long long) tr8->blck_size,  128);
        check_eq_i("traits type_size",  (long long) tr8->type_size,  130);
        check_eq_i("traits is_quantized", tr8->is_quantized ? 1 : 0, 1);
        check_eq_i("traits from_float_ref set", tr8->from_float_ref != NULL, 1);
        check_eq_i("traits to_float set",       tr8->to_float != NULL,       1);
        /* Turbo types drop the _0 suffix in their public name (turbo2/3/4/8),
         * exactly as in buun. ggml_type_name() is the accessor the CLI uses. */
        check_eq_i("ggml_type_name is \"turbo8\"",
                   strcmp(ggml_type_name(GGML_TYPE_TURBO8_0), "turbo8") == 0 ? 1 : 0, 1);
        check_eq_i("traits type_name agrees with ggml_type_name",
                   strcmp(tr8->type_name, ggml_type_name(GGML_TYPE_TURBO8_0)) == 0 ? 1 : 0, 1);
    }
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 1: turbo3 basis vector
     * --------------------------------------------------------------- */
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, store.bytes, d);
    dequantize_row_turbo3_0(store.bytes, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    print_vec("In: ", input);
    print_vec("Out:", output);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    printf("  MSE=%.8f Cosine=%.6f OutNorm=%.6f\n", mse, cosv, out_norm);
    check_ge("turbo3 e0 cosine", cosv, 0.99);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 2: turbo3 large-norm vector
     * --------------------------------------------------------------- */
    for (int i = 0; i < d; i++) { input[i] = sinf(i * 0.1f + 0.5f) * 10.0f; }
    quantize_row_turbo3_0_ref(input, store.bytes, d);
    dequantize_row_turbo3_0(store.bytes, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 2 (turbo3): sin*10\n");
    print_vec("In: ", input);
    print_vec("Out:", output);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    printf("  MSE=%.8f Cosine=%.6f InNorm=%.2f OutNorm=%.2f\n", mse, cosv, in_norm, out_norm);
    /* 0.98 is a regression floor for the 3-bit codec, not a quality target:
     * master's turbo3 measures ~0.986 here. */
    check_ge("turbo3 sin cosine", cosv, 0.98);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 3: turbo4
     * --------------------------------------------------------------- */
    for (int i = 0; i < d; i++) { input[i] = cosf(i * 0.2f) * 5.0f; }
    quantize_row_turbo4_0_ref(input, store.bytes, d);
    dequantize_row_turbo4_0(store.bytes, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 3 (turbo4): cos*5\n");
    print_vec("In: ", input);
    print_vec("Out:", output);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    printf("  MSE=%.8f Cosine=%.6f\n", mse, cosv);
    /* regression floor for the 4-bit codec (master measures ~0.989 here) */
    check_ge("turbo4 cos cosine", cosv, 0.98);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 4: turbo8 smooth signal
     * --------------------------------------------------------------- */
    for (int i = 0; i < d; i++) { input[i] = sinf(i * 0.37f + 1.1f) * 7.0f; }
    roundtrip8(input, output, d, store.bytes);
    printf("Test 4 (turbo8): sin*7\n");
    print_vec("In: ", input);
    print_vec("Out:", output);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    printf("  MSE=%.3e Cosine=%.8f InNorm=%.4f OutNorm=%.4f\n", mse, cosv, in_norm, out_norm);
    check_ge("turbo8 smooth cosine", cosv, 0.9999);
    check_le("turbo8 smooth rel-MSE", mse / (in_norm * in_norm / d), 1e-3);
    check_le("turbo8 smooth |norm ratio - 1|", fabs(out_norm / in_norm - 1.0), 0.02);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 5: turbo8 pseudo-random block
     * --------------------------------------------------------------- */
    for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }
    roundtrip8(input, output, d, store.bytes);
    printf("Test 5 (turbo8): pseudo-random\n");
    print_vec("In: ", input);
    print_vec("Out:", output);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    printf("  MSE=%.3e Cosine=%.8f\n", mse, cosv);
    check_ge("turbo8 random cosine", cosv, 0.999);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 6: turbo8 multi-block row -- per-block independence
     * (a group-size / block-boundary bug would show up as one bad block)
     * --------------------------------------------------------------- */
    {
        const int n6 = 4 * QK_TURBO8;
        for (int i = 0; i < n6; i++) { input[i] = rnd() * 2.0f; }
        roundtrip8(input, output, n6, store.bytes);
        printf("Test 6 (turbo8): 4-block row of %d\n", n6);
        stats(input, output, n6, &mse, &cosv, &in_norm, &out_norm);
        printf("  MSE=%.3e Cosine=%.8f\n", mse, cosv);
        check_ge("turbo8 row cosine", cosv, 0.999);
        for (int b = 0; b < n6 / QK_TURBO8; b++) {
            double bmse, bcos, bni, bno;
            char name[64];
            stats(input + b * QK_TURBO8, output + b * QK_TURBO8, QK_TURBO8, &bmse, &bcos, &bni, &bno);
            snprintf(name, sizeof(name), "turbo8 block %d cosine", b);
            check_ge(name, bcos, 0.999);
        }
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 7: turbo8 zero vector stays zero
     * --------------------------------------------------------------- */
    memset(input, 0, sizeof(float) * d);
    roundtrip8(input, output, d, store.bytes);
    {
        double mx = 0.0;
        for (int i = 0; i < d; i++) { mx = fmax(mx, fabs((double) output[i])); }
        printf("Test 7 (turbo8): zero vector\n");
        check_le("turbo8 zero max |out|", mx, 0.0);
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 8: turbo8 basis vector -- must land on grid corners (near exact)
     * --------------------------------------------------------------- */
    memset(input, 0, sizeof(float) * d);
    input[3] = 2.5f;
    roundtrip8(input, output, d, store.bytes);
    printf("Test 8 (turbo8): 2.5 * e3\n");
    print_vec("In: ", input);
    print_vec("Out:", output);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    {
        double mx = 0.0;
        for (int i = 0; i < d; i++) { mx = fmax(mx, fabs((double) input[i] - (double) output[i])); }
        printf("  Cosine=%.8f max abs err=%.3e\n", cosv, mx);
        check_ge("turbo8 basis cosine", cosv, 0.99999);
        check_le("turbo8 basis max abs err", mx, 1e-3);
    }
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 9: turbo8 norm preservation on a large-norm vector
     * --------------------------------------------------------------- */
    for (int i = 0; i < d; i++) { input[i] = cosf(i * 0.21f) * 100.0f; }
    roundtrip8(input, output, d, store.bytes);
    stats(input, output, d, &mse, &cosv, &in_norm, &out_norm);
    printf("Test 9 (turbo8): cos*100, |out|/|in| = %.6f\n", out_norm / in_norm);
    check_le("turbo8 large-norm |ratio - 1|", fabs(out_norm / in_norm - 1.0), 0.01);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 10: decoder format -- y[i] = ((qs[i] - 127.5) / 127.5) * norm
     * Hand-built block with norm = fp16(1.0) so the expected values are exact.
     * --------------------------------------------------------------- */
    {
        blk_turbo8_mirror * blk = &store.b[0];
        memset(blk, 0, sizeof(*blk));
        blk->norm = 0x3C00; /* fp16 1.0 */
        for (int i = 0; i < QK_TURBO8; i++) { blk->qs[i] = (unsigned char) i; }
        blk->qs[1] = 255;
        blk->qs[2] = 0;
        dequantize_row_turbo8_0(blk, output, QK_TURBO8);

        double fmt_err = 0.0;
        for (int i = 0; i < QK_TURBO8; i++) {
            const double want = ((double) blk->qs[i] - 127.5) / 127.5;
            const double e    = fabs((double) output[i] - want);
            if (e > fmt_err) { fmt_err = e; }
        }
        printf("Test 10 (turbo8): decoder format\n");
        check_le("turbo8 decoder format max abs err", fmt_err, 1e-6);
        check_le("turbo8 qs=0   -> -1.0", fabs((double) output[2] + 1.0), 1e-6);
        check_le("turbo8 qs=255 -> +1.0", fabs((double) output[1] - 1.0), 1e-6);
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 11: ggml_quantize_chunk() dispatch == the ref encoder
     * (this is the path llama-quantize / imatrix-aware quantization uses)
     * --------------------------------------------------------------- */
    {
        static unsigned char disp[3 * 130];
        const int nrows = 3, nper = QK_TURBO8;
        for (int i = 0; i < nrows * nper; i++) { input[i] = rnd() * 1.5f; }

        const size_t got = ggml_quantize_chunk(GGML_TYPE_TURBO8_0, input, disp, 0, nrows, nper, NULL);
        printf("Test 11 (turbo8): ggml_quantize_chunk dispatch\n");
        check_eq_i("dispatched byte count", (long long) got, (long long) (nrows * 130));

        int same = 1;
        for (int r = 0; r < nrows; r++) {
            unsigned char ref[130];
            quantize_row_turbo8_0_ref(input + r * nper, ref, nper);
            if (memcmp(ref, disp + (size_t) r * 130, 130) != 0) { same = 0; }
        }
        check_eq_i("dispatch output identical to ref encoder", same, 1);
        printf("\n");
    }

    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
