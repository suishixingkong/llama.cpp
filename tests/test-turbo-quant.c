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
extern void quantize_row_turbo2_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo2_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void quantize_row_turbo8_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo8_0(const void * x, float * y, long long k);
extern void turbo_cpu_fwht(float * x, int group_size);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

extern void quantize_row_turbo3_tcq_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_tcq(const void * x, float * y, long long k);
extern void quantize_row_turbo2_tcq_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo2_tcq(const void * x, float * y, long long k);
extern void quantize_row_turbo1_tcq_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo1_tcq(const void * x, float * y, long long k);

/* q8_0 reference codec, for the turbo8-vs-q8_0 quality comparison (both 8-bit). */
extern void quantize_row_q8_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_q8_0(const void * x, float * y, long long k);

/* The TCQ trellis codebooks, taken from the very same data file the library compiles in,
 * so the test can verify the encoder's chosen path cost without inventing a second
 * source of truth for the quantizer. */
#define TURBO_TCQ_TABLE static const
#include "ggml-turbo-tcq-tables.h"

#define QK_TURBO8   128
#define MAX_BLOCKS  8

/* Mirrors block_turbo8_0 from ggml-common.h: fp16 norm + 128 one-byte indices. */
typedef struct {
    unsigned short norm;
    unsigned char  qs[QK_TURBO8];
} blk_turbo8_mirror;

/* Mirror the three TCQ block layouts from ggml-common.h. */
typedef struct { unsigned short norm; unsigned char qs[49]; unsigned char pad; } blk_t3tcq_mirror; /* 52 B */
typedef struct { unsigned short norm; unsigned char qs[33]; unsigned char pad; } blk_t2tcq_mirror; /* 36 B */
typedef struct { unsigned short norm; unsigned char qs[17]; unsigned char pad; } blk_t1tcq_mirror; /* 20 B */

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

/* ---------- TCQ (trellis-coded quantization) helpers ---------- */

typedef void (*tcq_quant_fn)  (const float * x, void * y, long long k);
typedef void (*tcq_dequant_fn)(const void  * x, float * y, long long k);

/* Sliding nbits-wide window of the trellis bitstream at element t (LSB-first). */
static int tcq_window(const unsigned char * qs, int t, int k, int nbits) {
    const int bit_pos  = t * k;
    const int byte_idx = bit_pos >> 3;
    const int bit_off  = bit_pos & 7;
    const unsigned int raw = (unsigned int) qs[byte_idx] | ((unsigned int) qs[byte_idx + 1] << 8);
    return (int) ((raw >> bit_off) & ((1u << nbits) - 1u));
}

/* Cost of the path the encoder actually wrote, replayed from its own bitstream. */
static double tcq_written_path_cost(const unsigned char * qs, const float * xrot,
                                    const float * cb, int k, int w) {
    const int shift = k * (w - 1);
    int init_bits = 0;
    for (int pos = 0; pos < shift; pos++) {
        init_bits |= ((qs[pos >> 3] >> (pos & 7)) & 1) << pos;
    }
    /* The pre-group state's low k bits never reach an emitted state, so zero them. */
    int cur = init_bits << k;
    double total = 0.0;
    for (int t = 0; t < 128; t++) {
        const int sym = tcq_window(qs, t, k, k * w) >> shift;
        cur = (cur >> k) | (sym << shift);
        const double d = (double) xrot[t] - (double) cb[cur];
        total += d * d;
    }
    return total;
}

/* Independent optimum: enumerate (prev, sym) forwards instead of enumerating each
 * state's 2^k predecessors, so a bug in the encoder's grouping cannot hide. */
static double tcq_optimal_path_cost(const float * xrot, const float * cb, int k, int w) {
    const int shift = k * (w - 1);
    const int nn    = 1 << (k * w);
    static double cost[512], ncost[512];

    for (int s = 0; s < nn; s++) { cost[s] = 0.0; }
    for (int t = 0; t < 128; t++) {
        for (int s = 0; s < nn; s++) { ncost[s] = 1e300; }
        for (int prev = 0; prev < nn; prev++) {
            const double c0 = cost[prev];
            for (int sym = 0; sym < (1 << k); sym++) {
                const int ns   = (prev >> k) | (sym << shift);
                const double d = (double) xrot[t] - (double) cb[ns];
                const double c = c0 + d * d;
                if (c < ncost[ns]) { ncost[ns] = c; }
            }
        }
        for (int s = 0; s < nn; s++) { cost[s] = ncost[s]; }
    }
    double best = 1e300;
    for (int s = 0; s < nn; s++) { if (cost[s] < best) { best = cost[s]; } }
    return best;
}

/* Rotate a 128-element group exactly like the reference encoders do. */
static void tcq_rotate_group(const float * in, float * out, double * grp_norm) {
    double ns = 0.0;
    for (int i = 0; i < 128; i++) { out[i] = in[i]; ns += (double) in[i] * (double) in[i]; }
    const float gn  = (float) sqrt(ns);
    const float inv = (gn > 1e-10f) ? 1.0f / gn : 0.0f;
    for (int i = 0; i < 128; i++) { out[i] *= inv; }
    turbo_cpu_fwht(out, 128);
    *grp_norm = (double) gn;
}

static union {
    blk_t3tcq_mirror b3;
    blk_t2tcq_mirror b2;
    blk_t1tcq_mirror b1;
    unsigned char    bytes[52];
} tstore;

/* Constructive bitstream test: build a block from a KNOWN symbol sequence and check the
 * decoder reproduces cb[state_t] for the state_t the trellis definition implies. */
static void tcq_check_layout(const char * name, int k, int w, int nbytes,
                             const float * cb, tcq_dequant_fn deq) {
    const int shift = k * (w - 1);
    const int nsym  = 1 << k;
    /* Force the low k bits to 1: that is the part of the pre-group state the bitstream
     * cannot represent, so it must not influence any decoded element. */
    const int init_state = (0x15A % (1 << (k * w))) | ((1 << k) - 1);

    memset(&tstore, 0, sizeof(tstore));
    tstore.b3.norm = 0x3C00;                          /* fp16 1.0 */
    unsigned char * qs = tstore.bytes + 2;            /* bitstream starts after the norm */

    const int init_bits = (init_state >> k) & ((1 << shift) - 1);
    for (int pos = 0; pos < shift; pos++) {
        if ((init_bits >> pos) & 1) { qs[pos >> 3] |= (unsigned char) (1u << (pos & 7)); }
    }
    int sym[128];
    for (int t = 0; t < 128; t++) {
        sym[t] = t % nsym;
        for (int b = 0; b < k; b++) {
            const int pos = shift + t * k + b;
            if ((sym[t] >> b) & 1) { qs[pos >> 3] |= (unsigned char) (1u << (pos & 7)); }
        }
    }
    (void) nbytes;

    float dec[128];
    deq(&tstore, dec, 128);

    int cur = init_state, wrong = 0;
    double maxerr = 0.0;
    for (int t = 0; t < 128; t++) {
        cur = (cur >> k) | (sym[t] << shift);
        const double e = fabs((double) dec[t] - (double) cb[cur]);
        if (e > maxerr) { maxerr = e; }
        if (e > 1e-6)   { wrong++; }
    }
    char nm[96];
    snprintf(nm, sizeof nm, "%s layout: elements disagreeing with the trellis definition", name);
    check_eq_i(nm, wrong, 0);
    snprintf(nm, sizeof nm, "%s layout max abs err", name);
    check_le(nm, maxerr, 1e-6);
}

/* End-to-end: encode -> decode -> inverse FWHT -> compare with the input, plus a
 * proof that the encoder's path cost equals the independent DP optimum. */
static void tcq_full_case(const char * name, int k, int w,
                          const float * cb, tcq_quant_fn q, tcq_dequant_fn deq,
                          const float * in, double cos_min) {
    static float dec[128], xrot[128];
    double mse, cos, in_norm, out_norm, grp_norm, wcost, ocost;

    memset(&tstore, 0, sizeof(tstore));
    q(in, &tstore, 128);
    deq(&tstore, dec, 128);
    turbo_cpu_fwht_inverse(dec, 128);           /* decode stays rotated; undo for comparison */
    stats(in, dec, 128, &mse, &cos, &in_norm, &out_norm);

    tcq_rotate_group(in, xrot, &grp_norm);
    wcost = tcq_written_path_cost(tstore.bytes + 2, xrot, cb, k, w);
    ocost = tcq_optimal_path_cost(xrot, cb, k, w);

    printf("  %s: cosine=%.6f |out|/|in|=%.6f  path cost=%.6e optimum=%.6e\n",
           name, cos, out_norm / in_norm, wcost, ocost);

    char nm[96];
    snprintf(nm, sizeof nm, "%s round-trip cosine", name);
    check_ge(nm, cos, cos_min);
    snprintf(nm, sizeof nm, "%s |out|/|in| - 1", name);
    check_le(nm, fabs(out_norm / in_norm - 1.0), 0.02);
    snprintf(nm, sizeof nm, "%s encoder path cost == DP optimum (rel)", name);
    check_le(nm, fabs(wcost - ocost) / ocost, 1e-6);
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

    /* ---------------------------------------------------------------
     * TCQ (trellis-coded quantization) codecs
     * --------------------------------------------------------------- */
    printf("--- TurboQuant TCQ trellis codecs ---\n\n");

    printf("Test 12: TCQ block layout & registration\n");
    check_eq_i("sizeof(block_turbo3_tcq) mirror", (long long) sizeof(blk_t3tcq_mirror), 52);
    check_eq_i("sizeof(block_turbo2_tcq) mirror", (long long) sizeof(blk_t2tcq_mirror), 36);
    check_eq_i("sizeof(block_turbo1_tcq) mirror", (long long) sizeof(blk_t1tcq_mirror), 20);
    {
        const struct { enum ggml_type t; const char * name; long long tsize; } tcqs[3] = {
            { GGML_TYPE_TURBO3_TCQ, "turbo3_tcq", 52 },
            { GGML_TYPE_TURBO2_TCQ, "turbo2_tcq", 36 },
            { GGML_TYPE_TURBO1_TCQ, "turbo1_tcq", 20 },
        };
        for (int i = 0; i < 3; i++) {
            char nm[80];
            const struct ggml_type_traits * tr = ggml_get_type_traits(tcqs[i].t);
            if (tr == NULL) {
                g_checks++; g_fail++;
                printf("  [FAIL] ggml_get_type_traits(%s) returned NULL\n", tcqs[i].name);
                continue;
            }
            snprintf(nm, sizeof nm, "%s blck_size", tcqs[i].name);
            check_eq_i(nm, (long long) tr->blck_size, 128);
            snprintf(nm, sizeof nm, "%s type_size", tcqs[i].name);
            check_eq_i(nm, (long long) tr->type_size, tcqs[i].tsize);
            snprintf(nm, sizeof nm, "%s public name", tcqs[i].name);
            check_eq_i(nm, strcmp(ggml_type_name(tcqs[i].t), tcqs[i].name) == 0 ? 1 : 0, 1);
            snprintf(nm, sizeof nm, "%s codec hooks set", tcqs[i].name);
            check_eq_i(nm, (tr->from_float_ref != NULL && tr->to_float != NULL) ? 1 : 0, 1);
        }
    }
    printf("\n");

    printf("Test 13: TCQ bitstream layout (constructive, from known symbols)\n");
    tcq_check_layout("t3_tcq", 3, 3, 52, TURBO_TCQ3_CODEBOOK, dequantize_row_turbo3_tcq);
    tcq_check_layout("t2_tcq", 2, 4, 36, TURBO_TCQ2_CODEBOOK, dequantize_row_turbo2_tcq);
    tcq_check_layout("t1_tcq", 1, 8, 20, TURBO_TCQ1_CODEBOOK, dequantize_row_turbo1_tcq);
    printf("\n");

    printf("Test 14: TCQ3 (3.25 bpv) encode/decode + Viterbi optimality\n");
    for (int i = 0; i < d; i++) { input[i] = sinf(i * 0.37f + 1.1f) * 7.0f; }
    tcq_full_case("t3_tcq smooth", 3, 3, TURBO_TCQ3_CODEBOOK,
                  quantize_row_turbo3_tcq_ref, dequantize_row_turbo3_tcq, input, 0.98);
    for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }
    tcq_full_case("t3_tcq random", 3, 3, TURBO_TCQ3_CODEBOOK,
                  quantize_row_turbo3_tcq_ref, dequantize_row_turbo3_tcq, input, 0.98);
    printf("\n");

    printf("Test 15: TCQ2 (2.25 bpv) encode/decode + Viterbi optimality\n");
    for (int i = 0; i < d; i++) { input[i] = sinf(i * 0.37f + 1.1f) * 7.0f; }
    tcq_full_case("t2_tcq smooth", 2, 4, TURBO_TCQ2_CODEBOOK,
                  quantize_row_turbo2_tcq_ref, dequantize_row_turbo2_tcq, input, 0.95);
    for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }
    tcq_full_case("t2_tcq random", 2, 4, TURBO_TCQ2_CODEBOOK,
                  quantize_row_turbo2_tcq_ref, dequantize_row_turbo2_tcq, input, 0.95);
    printf("\n");

    printf("Test 15b: TCQ1 (1.25 bpv) encode/decode + Viterbi optimality\n");
    for (int i = 0; i < d; i++) { input[i] = sinf(i * 0.37f + 1.1f) * 7.0f; }
    tcq_full_case("t1_tcq smooth", 1, 8, TURBO_TCQ1_CODEBOOK,
                  quantize_row_turbo1_tcq_ref, dequantize_row_turbo1_tcq, input, 0.70);
    for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }
    tcq_full_case("t1_tcq random", 1, 8, TURBO_TCQ1_CODEBOOK,
                  quantize_row_turbo1_tcq_ref, dequantize_row_turbo1_tcq, input, 0.70);
    printf("\n");

    printf("Test 16: TCQ edge cases\n");
    {
        static float zdec[128];
        memset(input, 0, sizeof(float) * d);
        memset(&tstore, 0, sizeof(tstore));
        quantize_row_turbo3_tcq_ref(input, &tstore, d);
        dequantize_row_turbo3_tcq(&tstore, zdec, d);
        double mx = 0.0;
        for (int i = 0; i < d; i++) { mx = fmax(mx, fabs((double) zdec[i])); }
        check_le("t3_tcq zero vector max |out|", mx, 0.0);

        /* multi-block row: three groups must not interfere */
        static float mout[384];
        for (int i = 0; i < 384; i++) { input[i] = rnd() * 2.0f; }
        memset(&tstore, 0, sizeof(tstore));
        quantize_row_turbo3_tcq_ref(input, &tstore, 384);
        dequantize_row_turbo3_tcq(&tstore, mout, 384);
        for (int g = 0; g < 3; g++) { turbo_cpu_fwht_inverse(mout + g * 128, 128); }
        stats(input, mout, 384, &mse, &cosv, &in_norm, &out_norm);
        check_ge("t3_tcq 3-block row cosine", cosv, 0.98);
    }
    printf("\n");

    printf("Test 17: TCQ ggml_quantize_chunk dispatch\n");
    {
        static unsigned char disp[3 * 52];
        const int nrows = 3, nper = 128;
        for (int i = 0; i < nrows * nper; i++) { input[i] = rnd() * 1.5f; }

        const size_t got3 = ggml_quantize_chunk(GGML_TYPE_TURBO3_TCQ, input, disp, 0, nrows, nper, NULL);
        check_eq_i("t3_tcq dispatched byte count", (long long) got3, (long long) (nrows * 52));
        int same = 1;
        for (int r = 0; r < nrows; r++) {
            memset(&tstore, 0, sizeof(tstore));
            quantize_row_turbo3_tcq_ref(input + r * nper, &tstore, nper);
            if (memcmp(&tstore, disp + (size_t) r * 52, 52) != 0) { same = 0; }
        }
        check_eq_i("t3_tcq dispatch output identical to ref encoder", same, 1);

        const size_t got2 = ggml_quantize_chunk(GGML_TYPE_TURBO2_TCQ, input, disp, 0, nrows, nper, NULL);
        check_eq_i("t2_tcq dispatched byte count", (long long) got2, (long long) (nrows * 36));
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 18: the TCQ codecs must beat the scalar codec of the same bit tier on
     * identical data. This is buun's headline claim (its scalar turbo3 median KLD
     * 0.00279 vs 0.00163 for turbo3_tcq) expressed as something a CPU test can check.
     * Bit cost: turbo3_0 50 B vs turbo3_tcq 52 B; turbo2_0 34 B vs turbo2_tcq 36 B.
     * --------------------------------------------------------------- */
    printf("Test 18: TCQ vs the scalar codec on identical data\n");
    {
        static float sc[128], tc[128];
        double m1, c1, n1, o1, m2, c2, n2, o2;

        for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }

        memset(&tstore, 0, sizeof(tstore));
        quantize_row_turbo3_0_ref(input, &tstore, d);
        dequantize_row_turbo3_0(&tstore, sc, d);
        turbo_cpu_fwht_inverse(sc, d);
        stats(input, sc, d, &m1, &c1, &n1, &o1);

        memset(&tstore, 0, sizeof(tstore));
        quantize_row_turbo3_tcq_ref(input, &tstore, d);
        dequantize_row_turbo3_tcq(&tstore, tc, d);
        turbo_cpu_fwht_inverse(tc, d);
        stats(input, tc, d, &m2, &c2, &n2, &o2);

        printf("  turbo3_0   (50 B/128, scalar): cosine=%.6f\n", c1);
        printf("  turbo3_tcq (52 B/128, trellis): cosine=%.6f\n", c2);
        check_eq_i("t3_tcq cosine beats scalar turbo3_0", c2 > c1 ? 1 : 0, 1);

        memset(&tstore, 0, sizeof(tstore));
        quantize_row_turbo2_0_ref(input, &tstore, d);
        dequantize_row_turbo2_0(&tstore, sc, d);
        turbo_cpu_fwht_inverse(sc, d);
        stats(input, sc, d, &m1, &c1, &n1, &o1);

        memset(&tstore, 0, sizeof(tstore));
        quantize_row_turbo2_tcq_ref(input, &tstore, d);
        dequantize_row_turbo2_tcq(&tstore, tc, d);
        turbo_cpu_fwht_inverse(tc, d);
        stats(input, tc, d, &m2, &c2, &n2, &o2);

        printf("  turbo2_0   (34 B/128, scalar): cosine=%.6f\n", c1);
        printf("  turbo2_tcq (36 B/128, trellis): cosine=%.6f\n", c2);
        check_eq_i("t2_tcq cosine beats scalar turbo2_0", c2 > c1 ? 1 : 0, 1);
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 19: within the TCQ family, more bits per value must give a better
     * reconstruction. There is no scalar 1-bit codec in this tree to compare
     * against (buun's reserved turbo1 / turbo1_nsn / turbo1_cq slots are dead
     * codecs), so the check is that fidelity is monotone in the bit rate.
     * --------------------------------------------------------------- */
    printf("Test 19: TCQ family fidelity is monotone in the bit rate\n");
    {
        static float dec[128];
        double mse, cos, in_norm, out_norm;
        double cs[3];
        const struct {
            const char * name; int k, w, bytes;
            const float * cb; tcq_quant_fn q; tcq_dequant_fn dq;
        } fam[3] = {
            { "t1_tcq", 1, 8, 20, TURBO_TCQ1_CODEBOOK, quantize_row_turbo1_tcq_ref, dequantize_row_turbo1_tcq },
            { "t2_tcq", 2, 4, 36, TURBO_TCQ2_CODEBOOK, quantize_row_turbo2_tcq_ref, dequantize_row_turbo2_tcq },
            { "t3_tcq", 3, 3, 52, TURBO_TCQ3_CODEBOOK, quantize_row_turbo3_tcq_ref, dequantize_row_turbo3_tcq },
        };
        for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }
        for (int f = 0; f < 3; f++) {
            memset(&tstore, 0, sizeof(tstore));
            fam[f].q(input, &tstore, d);
            fam[f].dq(&tstore, dec, d);
            turbo_cpu_fwht_inverse(dec, d);
            stats(input, dec, d, &mse, &cos, &in_norm, &out_norm);
            cs[f] = cos;
            printf("  %s (%2d B/128): cosine=%.6f\n", fam[f].name, fam[f].bytes, cos);
        }
        check_eq_i("t2_tcq fidelity beats t1_tcq", cs[1] > cs[0] ? 1 : 0, 1);
        check_eq_i("t3_tcq fidelity beats t2_tcq", cs[2] > cs[1] ? 1 : 0, 1);
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 20: turbo8 vs q8_0 on identical data. Both are 8-bit, so this is the
     * quality question at the same bit rate tier: turbo8 rotates first (WHT spreads
     * outliers across the block) but quantises a whole 128-element block with one
     * absmax scale; q8_0 does not rotate but uses a per-32 block scale, i.e. finer
     * scale granularity.
     * Bit cost: q8_0 = 34 B / 32 values = 8.5 bpv; turbo8 = 130 B / 128 = 8.125 bpv.
     * turbo8's decode stays in the rotated domain, so the inverse WHT is applied before
     * comparing against the input; q8_0 needs no rotation.
     * --------------------------------------------------------------- */
    printf("Test 20: turbo8 vs q8_0 reconstruction quality (identical data)\n");
    {
        static unsigned char q8buf[512];
        static float t8[128], q8[128];
        double m1, c1, n1, o1, m2, c2, n2, o2;
        double sum_t8 = 0.0, sum_q8 = 0.0;
        double mse_t8 = 0.0, mse_q8 = 0.0;
        const int ntrials = 32;

        for (int trial = 0; trial < ntrials; trial++) {
            for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }

            memset(&tstore, 0, sizeof(tstore));
            quantize_row_turbo8_0_ref(input, &tstore, d);
            dequantize_row_turbo8_0(&tstore, t8, d);
            turbo_cpu_fwht_inverse(t8, d);
            stats(input, t8, d, &m1, &c1, &n1, &o1);
            sum_t8 += c1; mse_t8 += m1 / (n1 * n1 / d);

            memset(q8buf, 0, sizeof(q8buf));
            quantize_row_q8_0_ref(input, q8buf, d);
            dequantize_row_q8_0(q8buf, q8, d);
            stats(input, q8, d, &m2, &c2, &n2, &o2);
            sum_q8 += c2; mse_q8 += m2 / (n2 * n2 / d);
        }
        sum_t8 /= ntrials; sum_q8 /= ntrials;
        mse_t8 /= ntrials; mse_q8 /= ntrials;

        printf("  turbo8 (8.125 bpv, WHT + per-block absmax): mean cosine=%.6f  rel-MSE=%.3e\n", sum_t8, mse_t8);
        printf("  q8_0   (8.5   bpv, no WHT, per-32 scale)  : mean cosine=%.6f  rel-MSE=%.3e\n", sum_q8, mse_q8);
        printf("  -> higher cosine / lower rel-MSE is better; bit rates are within 4.4%% of each other\n");
        check_ge("turbo8 mean cosine is sane", sum_t8, 0.99);
        check_ge("q8_0 mean cosine is sane", sum_q8, 0.99);
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test 21: measured bpv (VRAM) and reconstruction fidelity for every KV-cache-capable
     * type, on identical data. Uses ggml's own registered traits, so the number reported is
     * what the library actually does. Rotated-domain (turbo) types get the inverse WHT
     * before comparison; the others are compared in place.
     * --------------------------------------------------------------- */
    printf("Test 21: bpv (VRAM) vs measured fidelity, all KV types, identical data\n");
    {
        static const enum ggml_type kv_types[] = {
            GGML_TYPE_F16,
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q4_0,
            GGML_TYPE_TURBO8_0,
            GGML_TYPE_TURBO4_0,
            GGML_TYPE_TURBO3_0,
            GGML_TYPE_TURBO2_0,
            GGML_TYPE_TURBO3_TCQ,
            GGML_TYPE_TURBO2_TCQ,
            GGML_TYPE_TURBO1_TCQ,
        };
        enum { KV_N = (int) (sizeof(kv_types)/sizeof(kv_types[0])) };
        static unsigned char buf[4096];
        static float out[128];
        double bpv[KV_N], relmse[2][KV_N], bpv_sorted[KV_N];
        const int ntrials = 32;

        /* Two data distributions. "uniform" is well behaved and favours the codec with the
         * finer scale granularity; "outliers" reproduces the KV-cache reality of a few
         * high-magnitude channels (attention sinks / massive activations), which is exactly
         * what turbo's WHT rotation exists to spread. Reporting only the first would make
         * turbo look strictly worse than the standard quants, which is a property of the
         * distribution, not of the codec. */
        static const char * mode_name[2] = { "uniform", "outliers" };

        for (int t = 0; t < KV_N; t++) {
            bpv[t] = 8.0 * (double) ggml_type_size(kv_types[t]) / (double) ggml_blck_size(kv_types[t]);
        }

        for (int mode = 0; mode < 2; mode++) {
            for (int t = 0; t < KV_N; t++) {
                const struct ggml_type_traits * tr = ggml_get_type_traits(kv_types[t]);
                const int is_turbo =
                    kv_types[t] == GGML_TYPE_TURBO8_0   || kv_types[t] == GGML_TYPE_TURBO4_0 ||
                    kv_types[t] == GGML_TYPE_TURBO3_0   || kv_types[t] == GGML_TYPE_TURBO2_0 ||
                    kv_types[t] == GGML_TYPE_TURBO3_TCQ || kv_types[t] == GGML_TYPE_TURBO2_TCQ ||
                    kv_types[t] == GGML_TYPE_TURBO1_TCQ;

                double acc = 0.0;
                for (int trial = 0; trial < ntrials; trial++) {
                    double m, c, n, o;
                    for (int i = 0; i < d; i++) { input[i] = rnd() * 3.0f; }
                    if (mode == 1) {
                        /* a handful of high-magnitude channels, the rest small */
                        for (int i = 0; i < d; i++) {
                            const int is_outlier = (i < 2) || (i >= 61 && i < 66);
                            input[i] = is_outlier ? input[i] * 20.0f : input[i] * 0.15f;
                        }
                    }

                    memset(buf, 0, sizeof(buf));
                    tr->from_float_ref(input, buf, d);
                    tr->to_float(buf, out, d);
                    if (is_turbo) { turbo_cpu_fwht_inverse(out, d); }
                    stats(input, out, d, &m, &c, &n, &o);
                    acc += m / (n * n / d);
                }
                relmse[mode][t] = acc / ntrials;
            }
        }

        /* print sorted by bit rate, ascending */
        for (int t = 0; t < KV_N; t++) { bpv_sorted[t] = bpv[t]; }
        printf("  %-12s %8s %14s %14s\n", "type", "bpv", mode_name[0], mode_name[1]);
        for (int rank = 0; rank < KV_N; rank++) {
            int best = -1;
            for (int t = 0; t < KV_N; t++) {
                if (bpv_sorted[t] < 0.0) { continue; }
                if (best < 0 || bpv_sorted[t] < bpv_sorted[best]) { best = t; }
            }
            printf("  %-12s %8.3f %14.3e %14.3e\n",
                   ggml_type_name(kv_types[best]), bpv[best], relmse[0][best], relmse[1][best]);
            bpv_sorted[best] = -1.0;
        }
        printf("  (rel-MSE, lower = better fidelity; bpv x 128 = bytes per 128 values)\n");
        printf("\n");
    }

    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
