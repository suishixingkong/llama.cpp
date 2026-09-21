// Verifies that the CUDA SET_ROWS encoders for the TurboQuant KV types produce
// byte-identical output to the CPU reference codec in ggml-turbo-quant.c.
//
// This is the highest-value check available for that port: the CPU reference is the format
// definition (and is covered by test-turbo-quant on the CPU), while the CUDA kernels were
// only compile-verified on the machine they were ported on. One SET_ROWS of a few rows
// exercises normalize -> FWHT -> Viterbi -> renormalize -> bitpack end to end, so a
// divergence in any of those steps shows up as differing bytes.
//
// No model file and no attention are needed: ggml's SET_ROWS op is exercised directly.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "testing.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int64_t CACHE_WIDTH = 128*3;   // three blocks per row
constexpr int64_t CACHE_ROWS  = 8;
constexpr int64_t UPDATE_ROWS = 4;

// Runs SET_ROWS (f32 -> quantized KV type) on `backend` and returns the resulting cache bytes.
std::vector<uint8_t> run_set_rows(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t cache_buft,
        ggml_type cache_type) {
    constexpr size_t N_TENSORS = 16;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false);
    const ggml_init_params params{context_bytes, nullptr, true};

    ggml_context_ptr cache_ctx(ggml_init(params));
    ggml_context_ptr compute_ctx(ggml_init(params));
    GGML_ASSERT(cache_ctx && compute_ctx);

    ggml_tensor * cache   = ggml_new_tensor_2d(cache_ctx.get(), cache_type,  CACHE_WIDTH, CACHE_ROWS);
    ggml_tensor * values  = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, CACHE_WIDTH, UPDATE_ROWS);
    ggml_tensor * indices = ggml_new_tensor_1d(compute_ctx.get(), GGML_TYPE_I32, UPDATE_ROWS);
    ggml_tensor * updated = ggml_set_rows(compute_ctx.get(), cache, values, indices);

    ggml_backend_buffer_ptr cache_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(cache_ctx.get(), cache_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    GGML_ASSERT(cache_buffer && compute_buffer);
    ggml_backend_buffer_clear(cache_buffer.get(), 0);

    // Wide dynamic range plus an LCG, so the trellis sees far more than a smooth ramp: a pure
    // sine leaves most of the codebook and most state transitions unexercised.
    std::vector<float> source(CACHE_WIDTH*UPDATE_ROWS);
    uint32_t rng = 12345u;
    for (size_t i = 0; i < source.size(); ++i) {
        rng = rng*1664525u + 1013904223u;
        const float u = float(rng >> 8)*(1.0f/16777216.0f)*2.0f - 1.0f;
        source[i] = u*3.0f + 0.25f*std::sin(float(i)*0.05f);
    }
    const std::vector<int32_t> rows{0, 3, 5, 7};

    ggml_backend_tensor_set(values,  source.data(),  0, source.size()*sizeof(float));
    ggml_backend_tensor_set(indices, rows.data(),    0, rows.size()*sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, updated);
    GGML_ASSERT(ggml_backend_supports_op(backend, updated));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> result(ggml_nbytes(cache));
    ggml_backend_tensor_get(cache, result.data(), 0, result.size());
    return result;
}

} // namespace

int main() {
    testing t;

    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    if (!cuda_backend) {
        printf("no CUDA device available - skipping\n");
        return 0;
    }

    // Only the types whose BLOCK IS THE ROTATION GROUP are compared here. turbo2_0/3_0/4_0
    // quantize a 128-element rotation group into several smaller blocks, and the CPU path takes
    // that group size from dst->op_params, which ggml_set_rows does not set. Comparing those
    // would therefore compare two different group-size assumptions rather than two
    // implementations.
    const ggml_type types[] = {
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ,
    };

    ggml_backend_ptr cpu_backend(ggml_backend_cpu_init());
    GGML_ASSERT(cpu_backend);

    for (ggml_type type : types) {
        t.test(ggml_type_name(type), [&](testing & t) {
            const std::vector<uint8_t> cpu = run_set_rows(
                cpu_backend.get(),  ggml_backend_get_default_buffer_type(cpu_backend.get()),  type);
            const std::vector<uint8_t> cuda = run_set_rows(
                cuda_backend.get(), ggml_backend_get_default_buffer_type(cuda_backend.get()), type);

            t.assert_equal(cpu.size(), cuda.size());
            if (cpu.size() != cuda.size()) {
                return;
            }

            size_t n_diff = 0;
            size_t first  = (size_t) -1;
            for (size_t i = 0; i < cpu.size(); ++i) {
                if (cpu[i] != cuda[i]) {
                    ++n_diff;
                    if (first == (size_t) -1) { first = i; }
                }
            }
            if (n_diff != 0) {
                printf("    %zu of %zu bytes differ; first at byte %zu "
                       "(row %zu, byte-in-row %zu) cpu=0x%02x cuda=0x%02x\n",
                       n_diff, cpu.size(), first,
                       first / (size_t) (CACHE_WIDTH*ggml_type_size(type)/ggml_blck_size(type)),
                       first % (size_t) (CACHE_WIDTH*ggml_type_size(type)/ggml_blck_size(type)),
                       cpu[first], cuda[first]);
            }
            t.assert_true("CUDA output is byte-identical to the CPU reference", n_diff == 0);
        });
    }
}
