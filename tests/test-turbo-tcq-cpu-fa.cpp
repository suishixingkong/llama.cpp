// End-to-end CPU-backend test for the TurboQuant TCQ KV cache types.
//
// This is the "can it actually be used?" test. It runs a real flash attention on the CPU
// backend with a TCQ KV cache, through the exact wiring llama-graph.cpp uses for turbo
// types:
//
//   K/V -> ggml_set_rows(TCQ)          (encoder: L2 norm -> FWHT -> Viterbi -> bitpack)
//   Q   -> ggml_turbo_wht(direction=0) (forward rotation)
//   attn-> ggml_flash_attn_ext
//   out -> ggml_turbo_wht(direction=1) (inverse rotation, V stays in the rotated domain)
//
// and compares against an unquantized f32 attention computed here. ggml's CPU flash
// attention is generic -- it needs type_traits_cpu[K].vec_dot and type_traits[V].to_float
// and nothing else -- so this also asserts that those registrations are what makes the
// types usable on CPU.
//
// No GPU needed: this runs on the CPU backend, which is why the numerics of the CPU codec
// and of the CPU attention path can be verified on a machine with no NVIDIA device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t D    = 128;  // head dim; TCQ blocks are 128 values
constexpr int64_t N_KV = 24;
constexpr int64_t N_Q  = 4;

int g_checks = 0;
int g_fail   = 0;

void check(bool ok, const char * what) {
    g_checks++;
    if (!ok) {
        g_fail++;
        printf("  FAIL: %s\n", what);
    }
}

// Deterministic pseudo-random, smooth-ish so it looks like real activation data.
float gen(size_t i) {
    return 0.5f*std::sin(float(i)*0.11f) + 0.3f*std::cos(float(i)*0.037f)
         + 0.2f*std::sin(float(i)*0.0071f + 1.3f);
}

struct ref_out {
    std::vector<float> data;  // [D * N_Q], index d + i*D
};

ref_out reference_attention(const float * K, const float * V, const float * Q, float scale) {
    ref_out r;
    r.data.assign(D*N_Q, 0.0f);

    std::vector<float> scores(N_KV);
    for (int64_t i = 0; i < N_Q; ++i) {
        float mx = -INFINITY;
        for (int64_t j = 0; j < N_KV; ++j) {
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                s += Q[d + i*D] * K[d + j*D];
            }
            scores[j] = s*scale;
            if (scores[j] > mx) { mx = scores[j]; }
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < N_KV; ++j) {
            scores[j] = std::exp(scores[j] - mx);
            sum += scores[j];
        }
        for (int64_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int64_t j = 0; j < N_KV; ++j) {
                acc += scores[j]/sum * V[d + j*D];
            }
            r.data[d + i*D] = acc;
        }
    }
    return r;
}

std::vector<float> run_turbo_attention(ggml_backend_t backend, ggml_type kv_type, float scale) {
    constexpr size_t N_TENSORS = 16;
    const size_t ctx_bytes = ggml_tensor_overhead()*N_TENSORS + ggml_graph_overhead_custom(N_TENSORS, false);
    const ggml_init_params ip{ctx_bytes, nullptr, true};

    ggml_context_ptr cache_ctx(ggml_init(ip));
    ggml_context_ptr comp_ctx(ggml_init(ip));
    GGML_ASSERT(cache_ctx && comp_ctx);

    // The KV cache lives in the cache context so it survives between calls (it is the
    // destination of SET_ROWS, i.e. written, not computed).
    ggml_tensor * k_cache = ggml_new_tensor_2d(cache_ctx.get(), kv_type, D, N_KV);
    ggml_tensor * v_cache = ggml_new_tensor_2d(cache_ctx.get(), kv_type, D, N_KV);

    ggml_tensor * k_src = ggml_new_tensor_2d(comp_ctx.get(), GGML_TYPE_F32, D, N_KV);
    ggml_tensor * v_src = ggml_new_tensor_2d(comp_ctx.get(), GGML_TYPE_F32, D, N_KV);
    ggml_tensor * idx   = ggml_new_tensor_1d(comp_ctx.get(), GGML_TYPE_I32, N_KV);
    ggml_tensor * q     = ggml_new_tensor_4d(comp_ctx.get(), GGML_TYPE_F32, D, N_Q, 1, 1);

    ggml_tensor * k_w = ggml_set_rows(comp_ctx.get(), k_cache, k_src, idx);
    ggml_tensor * v_w = ggml_set_rows(comp_ctx.get(), v_cache, v_src, idx);

    // Forward-rotate Q (direction 0), then inverse-rotate the attention output (direction 1),
    // exactly as llama-graph.cpp does for turbo KV types.
    ggml_tensor * q_rot = ggml_turbo_wht(comp_ctx.get(), q, 0, 0, nullptr);
    ggml_tensor * attn  = ggml_flash_attn_ext(comp_ctx.get(), q_rot, k_w, v_w, nullptr, scale, 0.0f, 0.0f);
    ggml_tensor * attn_c = ggml_is_contiguous(attn) ? attn : ggml_cont(comp_ctx.get(), attn);
    ggml_tensor * out   = ggml_turbo_wht(comp_ctx.get(), attn_c, 1, (int) D, nullptr);

    ggml_backend_buffer_ptr cache_buf(ggml_backend_alloc_ctx_tensors(cache_ctx.get(), backend));
    ggml_backend_buffer_ptr comp_buf(ggml_backend_alloc_ctx_tensors(comp_ctx.get(), backend));
    GGML_ASSERT(cache_buf && comp_buf);
    ggml_backend_buffer_clear(cache_buf.get(), 0);

    std::vector<float> k_data(D*N_KV), v_data(D*N_KV), q_data(D*N_Q);
    for (size_t i = 0; i < k_data.size(); ++i) { k_data[i] = gen(i); }
    for (size_t i = 0; i < v_data.size(); ++i) { v_data[i] = gen(i + 9173); }
    for (size_t i = 0; i < q_data.size(); ++i) { q_data[i] = gen(i + 5501); }
    std::vector<int32_t> rows(N_KV);
    for (int64_t j = 0; j < N_KV; ++j) { rows[j] = (int32_t) j; }

    ggml_backend_tensor_set(k_src, k_data.data(), 0, k_data.size()*sizeof(float));
    ggml_backend_tensor_set(v_src, v_data.data(), 0, v_data.size()*sizeof(float));
    ggml_backend_tensor_set(q,     q_data.data(), 0, q_data.size()*sizeof(float));
    ggml_backend_tensor_set(idx,   rows.data(),   0, rows.size()*sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph_custom(comp_ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, out);

    // The capability claim: the CPU backend must accept FLASH_ATTN_EXT with this KV type.
    check(ggml_backend_supports_op(backend, attn), "cpu backend supports FLASH_ATTN_EXT with TCQ KV");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute succeeds");

    std::vector<float> res(D*N_Q);
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));

    return res;
}

void cosine_of(const float * a, const float * b, int64_t n, double & cos_v, double & max_abs) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        dot += (double) a[i]*b[i];
        na  += (double) a[i]*a[i];
        nb  += (double) b[i]*b[i];
        double e = std::fabs((double) a[i] - b[i]);
        if (e > max_abs) { max_abs = e; }
    }
    cos_v = dot / (std::sqrt(na)*std::sqrt(nb) + 1e-30);
}

} // namespace

int main(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(dev != nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend != nullptr);

    const float scale = 1.0f / std::sqrt((float) D);

    // Reference data (regenerated identically inside each run).
    std::vector<float> k_data(D*N_KV), v_data(D*N_KV), q_data(D*N_Q);
    for (size_t i = 0; i < k_data.size(); ++i) { k_data[i] = gen(i); }
    for (size_t i = 0; i < v_data.size(); ++i) { v_data[i] = gen(i + 9173); }
    for (size_t i = 0; i < q_data.size(); ++i) { q_data[i] = gen(i + 5501); }
    ref_out ref = reference_attention(k_data.data(), v_data.data(), q_data.data(), scale);

    struct { ggml_type t; const char * name; double floor; } cases[] = {
        { GGML_TYPE_TURBO3_TCQ, "turbo3_tcq", 0.99 },
        { GGML_TYPE_TURBO2_TCQ, "turbo2_tcq", 0.97 },
        { GGML_TYPE_TURBO1_TCQ, "turbo1_tcq", 0.90 },
        { GGML_TYPE_TURBO8_0,   "turbo8",     0.999 },
        { GGML_TYPE_TURBO3_0,   "turbo3",     0.99 },
    };

    for (const auto & c : cases) {
        std::vector<float> got = run_turbo_attention(backend, c.t, scale);
        double cos_v, max_abs;
        cosine_of(got.data(), ref.data.data(), D*N_Q, cos_v, max_abs);
        printf("%-12s cosine=%.6f  max_abs_err=%.6f  (floor %.3f)\n", c.name, cos_v, max_abs, c.floor);
        check(std::isfinite(cos_v), "output is finite");
        check(max_abs < 10.0, "output magnitude is sane");
        char buf[128];
        snprintf(buf, sizeof(buf), "%s attention cosine >= %.3f", c.name, c.floor);
        check(cos_v >= c.floor, buf);
    }

    ggml_backend_free(backend);

    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
