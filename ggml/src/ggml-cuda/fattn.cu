#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-q8-volta.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "kv-stream-span-tuner.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const bool selected = i < ne30 && isfinite(__half2float(mask[i]));
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = row_count;
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, int(mask->ne[0]), n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2LL*n_kv_max);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}
struct ggml_cuda_kv_stream_resident_cache;

namespace {

constexpr size_t KV_STREAM_NO_REQUEST = std::numeric_limits<size_t>::max();
constexpr uint32_t KV_STREAM_NO_LAYER = std::numeric_limits<uint32_t>::max();
constexpr uint32_t KV_STREAM_COPY_BATCH_PAGES = 32;
constexpr uint32_t KV_STREAM_DECODE_SPAN_PAGES = KV_STREAM_COPY_BATCH_PAGES;

struct kv_stream_graph_request {
    const char * k_data = nullptr;
    const char * v_data = nullptr;
    size_t k_nb1 = 0;
    size_t k_nb2 = 0;
    size_t v_nb1 = 0;
    size_t v_nb2 = 0;
    int64_t n_head_kv = 0;
    int64_t token_begin = 0;
    int64_t token_count = 0;
    size_t k_row_bytes = 0;
    size_t v_row_bytes = 0;
    size_t k_head_bytes = 0;
    size_t k_bytes = 0;
    size_t v_offset = 0;
    size_t v_head_bytes = 0;
    size_t v_bytes = 0;
    uint32_t layer = 0;
    uint32_t slot = 0;
    uint32_t ready_slot = 0;
    bool mutable_tail = false;
    bool eligible = false;
    bool scheduled = false;
    bool consumed = false;
    bool deadline_sample = false;
};

} // namespace

struct ggml_cuda_kv_stream_transfer_ring {
    char * pool_data = nullptr;
    size_t page_bytes = 0;
    char * conversion_data = nullptr;
    size_t conversion_bytes = 0;
    uint32_t capacity_slots = 0;
    uint32_t active_slots = 0;
    uint32_t forced_decode_span_pages = 0;
    uint32_t graph_decode_span_pages = UINT32_MAX;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t eval_start = nullptr;
    cudaEvent_t eval_end = nullptr;
    cudaEvent_t copy_sample_start = nullptr;
    cudaEvent_t copy_sample_end = nullptr;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> consumed;
    std::vector<uint8_t> slot_used;
    std::vector<size_t> slot_request;
    uint32_t * ready_flags_host = nullptr;
    uint32_t * ready_flags_device = nullptr;
    uint64_t * deadline_counters_host = nullptr;
    uint64_t * deadline_counters_device = nullptr;

    bool graph_active = false;
    bool graph_decode = true;
    ggml_cuda_kv_stream_span_tuner span_tuner;
    uint32_t graph_layer_count = 0;
    uint32_t current_layer = KV_STREAM_NO_LAYER;
    size_t next_request = 0;
    ggml_cuda_kv_stream_resident_cache * graph_resident_cache = nullptr;
    std::vector<kv_stream_graph_request> graph_requests;
    std::unordered_map<const void *, uint32_t> graph_layer_by_k;
    std::unordered_map<const void *, std::vector<size_t>> graph_request_by_k_page;

    uint64_t asynchronous_page_uploads = 0;
    uint64_t host_to_device_copy_commands = 0;
    uint64_t compute_stream_waits = 0;
    uint64_t stage_slot_reuses = 0;
    uint64_t cross_layer_prefetches = 0;
    uint32_t current_occupancy = 0;
    uint32_t ring_peak_occupancy = 0;
    uint32_t current_ring_peak_occupancy = 0;
    uint32_t last_ring_peak_occupancy = 0;
    uint64_t current_epoch_uploads = 0;
    double last_copy_engine_busy_ratio = 0.0;
    uint32_t copy_sample_uploads = 0;
    bool timing_pending = false;
    bool timing_current = false;
    bool last_graph_decode = false;
    bool last_graph_bounded = false;
    bool last_graph_streamed = false;
    bool copy_sample_recorded = false;
};

ggml_cuda_kv_stream_transfer_ring * ggml_cuda_kv_stream_transfer_ring_new(
        void * pool_data, size_t page_bytes, uint32_t stage_slots,
        void * conversion_data, size_t conversion_bytes,
        uint32_t forced_decode_span_pages) {
    if (pool_data == nullptr || page_bytes == 0 || stage_slots == 0 ||
            (conversion_bytes != 0 && conversion_data == nullptr)) {
        return nullptr;
    }

    auto * ring = new ggml_cuda_kv_stream_transfer_ring;
    ring->pool_data = static_cast<char *>(pool_data);
    ring->page_bytes = page_bytes;
    ring->conversion_data = static_cast<char *>(conversion_data);
    ring->conversion_bytes = conversion_bytes;
    ring->capacity_slots = stage_slots;
    ring->active_slots = stage_slots;
    ring->ready.resize(stage_slots, nullptr);
    ring->forced_decode_span_pages = forced_decode_span_pages;
    ring->consumed.resize(stage_slots, nullptr);
    ring->slot_used.resize(stage_slots, 0);
    ring->slot_request.resize(stage_slots, KV_STREAM_NO_REQUEST);

    auto cleanup = [&]() {
        for (cudaEvent_t event : ring->ready) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        for (cudaEvent_t event : ring->consumed) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        if (ring->producer_ready != nullptr) {
            (void) cudaEventDestroy(ring->producer_ready);
        }
        if (ring->eval_start != nullptr) {
            (void) cudaEventDestroy(ring->eval_start);
        }
        if (ring->eval_end != nullptr) {
            (void) cudaEventDestroy(ring->eval_end);
        }
        if (ring->copy_sample_start != nullptr) {
            (void) cudaEventDestroy(ring->copy_sample_start);
        }
        if (ring->copy_sample_end != nullptr) {
            (void) cudaEventDestroy(ring->copy_sample_end);
        }
        if (ring->copy_stream != nullptr) {
            (void) cudaStreamDestroy(ring->copy_stream);
        }
        if (ring->ready_flags_host != nullptr) {
            (void) cudaFreeHost(ring->ready_flags_host);
        }
        if (ring->deadline_counters_host != nullptr) {
            (void) cudaFreeHost(ring->deadline_counters_host);
        }
        delete ring;
    };

    if (cudaHostAlloc(reinterpret_cast<void **>(&ring->ready_flags_host),
            stage_slots*sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring->ready_flags_device),
            ring->ready_flags_host, 0) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&ring->deadline_counters_host),
            2*sizeof(uint64_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring->deadline_counters_device),
            ring->deadline_counters_host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    std::fill_n(ring->ready_flags_host, stage_slots, 0u);
    std::fill_n(ring->deadline_counters_host, 2, uint64_t(0));

    if (cudaStreamCreateWithFlags(&ring->copy_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ring->producer_ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreate(&ring->eval_start) != cudaSuccess ||
        cudaEventCreate(&ring->eval_end) != cudaSuccess ||
        cudaEventCreate(&ring->copy_sample_start) != cudaSuccess ||
        cudaEventCreate(&ring->copy_sample_end) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    for (uint32_t slot = 0; slot < stage_slots; ++slot) {
        if (cudaEventCreateWithFlags(&ring->ready[slot], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ring->consumed[slot], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            cleanup();
            return nullptr;
        }
    }
    return ring;
}

void ggml_cuda_kv_stream_transfer_ring_free(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ring->copy_stream));
    for (cudaEvent_t event : ring->ready) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    for (cudaEvent_t event : ring->consumed) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    CUDA_CHECK(cudaEventDestroy(ring->producer_ready));
    CUDA_CHECK(cudaEventDestroy(ring->eval_start));
    CUDA_CHECK(cudaEventDestroy(ring->eval_end));
    CUDA_CHECK(cudaEventDestroy(ring->copy_sample_start));
    CUDA_CHECK(cudaEventDestroy(ring->copy_sample_end));
    CUDA_CHECK(cudaStreamDestroy(ring->copy_stream));
    CUDA_CHECK(cudaFreeHost(ring->ready_flags_host));
    CUDA_CHECK(cudaFreeHost(ring->deadline_counters_host));
    delete ring;
}

bool ggml_cuda_kv_stream_transfer_ring_set_active_slots(
        ggml_cuda_kv_stream_transfer_ring * ring, uint32_t stage_slots) {
    if (ring == nullptr || stage_slots == 0 || stage_slots > ring->capacity_slots) {
        return false;
    }
    if (ring->active_slots != stage_slots) {
        ring->span_tuner.reset();
        ring->timing_pending = false;
        ring->timing_current = false;
    }
    ring->active_slots = stage_slots;
    return true;
}

void ggml_cuda_kv_stream_transfer_ring_set_conversion_data(
        ggml_cuda_kv_stream_transfer_ring * ring, void * conversion_data) {
    if (ring != nullptr) {
        ring->conversion_data = static_cast<char *>(conversion_data);
    }
}

void ggml_cuda_kv_stream_transfer_ring_reset_span_tuner(
        ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring != nullptr) {
        ring->span_tuner.reset();
        ring->timing_pending = false;
        ring->timing_current = false;
    }
}

bool ggml_cuda_kv_stream_transfer_ring_observe_decode_latency(
        ggml_cuda_kv_stream_transfer_ring * ring, double elapsed_ms) {
    if (ring == nullptr) {
        return false;
    }
    if (ring->forced_decode_span_pages != 0) {
        return true;
    }
    const bool was_selected = ring->span_tuner.selected();
    ring->span_tuner.observe(
        elapsed_ms, ring->last_graph_decode && ring->last_graph_streamed,
        ring->last_graph_bounded);
    if (!was_selected && ring->span_tuner.selected()) {
        GGML_LOG_WARN(
            "%s: selected %s decode spans from end-to-end latency "
            "(unbounded %.3f ms, %u-page %.3f ms)\n",
            __func__, ring->span_tuner.use_bounded() ? "bounded" : "unbounded",
            ring->span_tuner.unbounded_average_ms(), KV_STREAM_DECODE_SPAN_PAGES,
            ring->span_tuner.bounded_average_ms());
    }
    return true;
}

ggml_cuda_kv_stream_transfer_stats ggml_cuda_kv_stream_transfer_ring_get_stats(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return {};
    }
    return {
        ring->asynchronous_page_uploads,
        ring->host_to_device_copy_commands,
        ring->compute_stream_waits,
        ring->stage_slot_reuses,
        ring->cross_layer_prefetches,
        ring->deadline_counters_host[0],
        ring->deadline_counters_host[1],
        ring->ring_peak_occupancy,
    };
}

struct ggml_cuda_kv_stream_resident_cache {
    char * pool_data = nullptr;
    size_t pool_bytes = 0;
    size_t scratch_bytes = 0;
    size_t page_bytes = 0;
    uint32_t layer_count = 0;
    uint32_t page_tokens = 0;
    uint32_t resident_pages_per_layer = 0;
    uint32_t decode_active_pages = 0;
    uint32_t next_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    std::unordered_map<const void *, uint32_t> layer_by_k;
    std::unordered_map<const void *, uint32_t> layer_by_data;
    std::unordered_map<const void *, void *> mirror_by_data;
    std::vector<uint8_t> loaded;
    std::vector<uint8_t> dirty;
    std::vector<uint8_t> precise_dirty_tracking;
    std::vector<int64_t> dirty_rows;
    std::vector<uint32_t> mutable_pages;
    bool all_pages_mutable = false;
    ggml_cuda_kv_stream_resident_stats stats;
};

static bool kv_stream_resident_cache_layout(
        const ggml_cuda_kv_stream_resident_cache * cache,
        size_t pool_bytes,
        size_t scratch_bytes,
        uint32_t decode_active_pages,
        uint32_t & resident_pages_per_layer,
        std::vector<uint32_t> & layer_pages,
        std::vector<size_t> & layer_offsets) {
    if (cache == nullptr || scratch_bytes > pool_bytes ||
            scratch_bytes%cache->page_bytes != 0) {
        return false;
    }
    const size_t resident_pages_total =
        (pool_bytes - scratch_bytes)/cache->page_bytes;
    const size_t pages_per_layer = resident_pages_total/cache->layer_count;
    if (pages_per_layer > UINT32_MAX ||
            pages_per_layer > std::numeric_limits<size_t>::max()/cache->layer_count) {
        return false;
    }

    resident_pages_per_layer = uint32_t(pages_per_layer);
    const size_t controlled_resident_pages = pages_per_layer*cache->layer_count;
    layer_pages.assign(cache->layer_count, resident_pages_per_layer);
    if (decode_active_pages > resident_pages_per_layer) {
        const uint64_t total_active_pages =
            uint64_t(decode_active_pages)*cache->layer_count;
        if (total_active_pages < controlled_resident_pages) {
            return false;
        }
        const uint64_t streamed_pages =
            total_active_pages - controlled_resident_pages;
        const uint64_t ring_pages = scratch_bytes/cache->page_bytes;
        if (ring_pages == 0 || streamed_pages == 0) {
            return false;
        }

        // Reduce the number of split-attention layers while each streamed
        // working set fits in both the shared ring and the active pages owned
        // by one layer. When even one split per model layer exceeds the ring,
        // that layer streams in multiple waves. Spread split layers across
        // model order so resident layers provide prefetch windows.
        const auto ceil_div = [](uint64_t numerator, uint64_t denominator) {
            return numerator/denominator + (numerator%denominator != 0);
        };
        const uint64_t splits_for_ring = ceil_div(streamed_pages, ring_pages);
        const uint64_t splits_for_layer_capacity =
            ceil_div(streamed_pages, decode_active_pages);
        const uint64_t ring_bounded_splits =
            std::min<uint64_t>(cache->layer_count, splits_for_ring);
        const uint64_t split_layers_wide =
            std::max(ring_bounded_splits, splits_for_layer_capacity);
        if (split_layers_wide == 0 || split_layers_wide > cache->layer_count) {
            return false;
        }
        const uint32_t split_layers = uint32_t(split_layers_wide);
        const uint64_t base_streamed = streamed_pages/split_layers;
        const uint64_t remainder = streamed_pages%split_layers;
        layer_pages.assign(cache->layer_count, decode_active_pages);
        for (uint32_t split = 0; split < split_layers; ++split) {
            const uint32_t layer = uint32_t(
                uint64_t(split)*cache->layer_count/split_layers);
            const uint64_t layer_streamed = base_streamed + (split < remainder ? 1 : 0);
            if (layer_streamed > decode_active_pages) {
                return false;
            }
            layer_pages[layer] = decode_active_pages - uint32_t(layer_streamed);
        }
    }

    layer_offsets.assign(size_t(cache->layer_count) + 1, 0);
    for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
        layer_offsets[layer + 1] = layer_offsets[layer] + layer_pages[layer];
    }
    return layer_offsets.back() == controlled_resident_pages;
}

static size_t kv_stream_resident_index(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t layer,
        uint32_t page) {
    GGML_ASSERT(layer < cache->layer_count && page < cache->layer_pages[layer]);
    return cache->layer_offsets[layer] + page;
}

ggml_cuda_kv_stream_resident_cache * ggml_cuda_kv_stream_resident_cache_new(
        void * pool_data, size_t pool_bytes, size_t scratch_bytes, size_t page_bytes,
        uint32_t layer_count, uint32_t page_tokens) {
    if (pool_data == nullptr || scratch_bytes == 0 || scratch_bytes >= pool_bytes ||
            page_bytes == 0 || scratch_bytes%page_bytes != 0 ||
            layer_count == 0 || page_tokens != 256) {
        return nullptr;
    }

    const size_t resident_pages = (pool_bytes - scratch_bytes)/(page_bytes*layer_count);
    if (resident_pages == 0 || resident_pages > UINT32_MAX) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_kv_stream_resident_cache;
    cache->pool_data = static_cast<char *>(pool_data);
    cache->pool_bytes = pool_bytes;
    cache->scratch_bytes = scratch_bytes;
    cache->page_bytes = page_bytes;
    cache->layer_count = layer_count;
    cache->page_tokens = page_tokens;
    cache->resident_pages_per_layer = resident_pages;
    cache->layer_pages.assign(layer_count, uint32_t(resident_pages));
    cache->layer_offsets.resize(size_t(layer_count) + 1);
    for (uint32_t layer = 0; layer <= layer_count; ++layer) {
        cache->layer_offsets[layer] = size_t(layer)*resident_pages;
    }
    cache->loaded.resize(cache->layer_offsets.back(), 0);
    cache->dirty.resize(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.resize(layer_count, 0);
    return cache;
}

void ggml_cuda_kv_stream_resident_cache_free(ggml_cuda_kv_stream_resident_cache * cache) {
    delete cache;
}

void ggml_cuda_kv_stream_resident_cache_reset(ggml_cuda_kv_stream_resident_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    std::fill(cache->loaded.begin(), cache->loaded.end(), 0);
    std::fill(cache->dirty.begin(), cache->dirty.end(), 0);
    std::fill(cache->precise_dirty_tracking.begin(), cache->precise_dirty_tracking.end(), 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->stats = {};
}

bool ggml_cuda_kv_stream_resident_cache_reconfigure(
        ggml_cuda_kv_stream_resident_cache * cache,
        size_t scratch_bytes,
        uint32_t active_pages_per_layer) {
    if (cache == nullptr) {
        return false;
    }

    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, cache->pool_bytes, scratch_bytes, active_pages_per_layer,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }

    const bool scratch_changed = cache->scratch_bytes != scratch_bytes;
    const bool layout_changed = cache->layer_pages != layer_pages;
    cache->decode_active_pages = active_pages_per_layer;
    cache->resident_pages_per_layer = pages_per_layer;
    if (!scratch_changed && !layout_changed) {
        return true;
    }

    // Publish the scratch boundary and concentrated decode layout together.
    // Both changes alter physical K/V addresses, so one invalidation is
    // required; applying them separately would discard and reload the same
    // resident working set twice.
    cache->scratch_bytes = scratch_bytes;
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    if (scratch_changed) {
        cache->stats = {};
    }
    return true;
}

bool ggml_cuda_kv_stream_resident_cache_resize(
        ggml_cuda_kv_stream_resident_cache * cache,
        size_t pool_bytes,
        size_t scratch_bytes,
        uint32_t active_pages_per_layer) {
    if (cache == nullptr) {
        return false;
    }

    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, pool_bytes, scratch_bytes, active_pages_per_layer,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }

    const bool pool_changed = cache->pool_bytes != pool_bytes;
    const bool scratch_changed = cache->scratch_bytes != scratch_bytes;
    const bool layout_changed = cache->layer_pages != layer_pages;
    cache->pool_bytes = pool_bytes;
    cache->decode_active_pages = active_pages_per_layer;
    cache->resident_pages_per_layer = pages_per_layer;
    if (!scratch_changed && !layout_changed) {
        return true;
    }

    cache->scratch_bytes = scratch_bytes;
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    if (pool_changed || scratch_changed) {
        cache->stats = {};
    }
    return true;
}


bool ggml_cuda_kv_stream_resident_cache_repartition(
        ggml_cuda_kv_stream_resident_cache * cache, size_t scratch_bytes) {
    if (cache == nullptr) {
        return false;
    }
    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, cache->pool_bytes, scratch_bytes, cache->decode_active_pages,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }
    if (cache->scratch_bytes == scratch_bytes && cache->layer_pages == layer_pages) {
        return true;
    }
    cache->scratch_bytes = scratch_bytes;
    cache->resident_pages_per_layer = pages_per_layer;
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->stats = {};
    return true;
}

bool ggml_cuda_kv_stream_resident_cache_set_decode_layout(
        ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t active_pages_per_layer) {
    if (cache == nullptr) {
        return false;
    }
    uint32_t pages_per_layer = 0;
    std::vector<uint32_t> layer_pages;
    std::vector<size_t> layer_offsets;
    if (!kv_stream_resident_cache_layout(
            cache, cache->pool_bytes, cache->scratch_bytes, active_pages_per_layer,
            pages_per_layer, layer_pages, layer_offsets)) {
        return false;
    }
    if (cache->layer_pages == layer_pages) {
        cache->decode_active_pages = active_pages_per_layer;
        return true;
    }
    // Resident storage uses separate contiguous K and V planes per layer.
    // A layout change moves both the layer base and the V-plane boundary, so
    // migrating page-sized byte slots cannot preserve logical K/V pages.
    // Reload lazily from the authoritative host cache instead.
    cache->layer_pages = std::move(layer_pages);
    cache->layer_offsets = std::move(layer_offsets);
    cache->loaded.assign(cache->layer_offsets.back(), 0);
    cache->dirty.assign(cache->layer_offsets.back(), 0);
    cache->precise_dirty_tracking.assign(cache->layer_count, 0);
    cache->dirty_rows.clear();
    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    cache->layer_by_k.clear();
    cache->layer_by_data.clear();
    cache->mirror_by_data.clear();
    cache->next_layer = 0;
    cache->decode_active_pages = active_pages_per_layer;
    cache->resident_pages_per_layer = pages_per_layer;
    return true;
}

uint32_t ggml_cuda_kv_stream_resident_cache_pages_per_layer(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->resident_pages_per_layer;
}

uint32_t ggml_cuda_kv_stream_resident_cache_decode_active_pages(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->decode_active_pages;
}

ggml_cuda_kv_stream_resident_stats ggml_cuda_kv_stream_resident_cache_get_stats(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? ggml_cuda_kv_stream_resident_stats{} : cache->stats;
}

bool ggml_cuda_kv_stream_resident_cache_mark_dirty_rows(
        ggml_cuda_kv_stream_resident_cache * cache,
        const int64_t * rows, size_t count) {
    if (cache == nullptr || (rows == nullptr && count != 0)) {
        return false;
    }

    cache->mutable_pages.clear();
    cache->all_pages_mutable = false;
    if (count == 0) {
        cache->dirty_rows.clear();
    } else {
        cache->dirty_rows.assign(rows, rows + count);
    }
    std::fill(cache->dirty.begin(), cache->dirty.end(), 0);
    for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
        cache->precise_dirty_tracking[layer] = 1;
    }
    for (size_t i = 0; i < count; ++i) {
        if (rows[i] < 0) {
            cache->all_pages_mutable = true;
            std::fill(cache->dirty.begin(), cache->dirty.end(), 1);
            return true;
        }
        const uint32_t page = uint32_t(uint64_t(rows[i])/cache->page_tokens);
        if (std::find(cache->mutable_pages.begin(), cache->mutable_pages.end(), page) ==
                cache->mutable_pages.end()) {
            cache->mutable_pages.push_back(page);
        }
        for (uint32_t layer = 0; layer < cache->layer_count; ++layer) {
            if (page < cache->layer_pages[layer]) {
                cache->dirty[kv_stream_resident_index(cache, layer, page)] = 1;
            }
        }
    }
    return true;
}

bool ggml_cuda_kv_stream_resident_cache_all_layers_fit(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t active_pages) {
    if (cache == nullptr || active_pages == 0) {
        return false;
    }
    return std::all_of(cache->layer_pages.begin(), cache->layer_pages.end(),
        [active_pages](uint32_t pages) { return pages >= active_pages; });
}

bool ggml_cuda_kv_stream_resident_cache_get_mirror(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target,
        void ** data) {
    if (cache == nullptr || target == nullptr || data == nullptr ||
            cache->dirty_rows.empty()) {
        return false;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    const auto mirror_it = cache->mirror_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end() || mirror_it == cache->mirror_by_data.end()) {
        return false;
    }
    const uint64_t capacity = uint64_t(cache->layer_pages[layer_it->second])*cache->page_tokens;
    for (const int64_t row : cache->dirty_rows) {
        if (row < 0 || uint64_t(row) >= capacity) {
            return false;
        }
    }
    *data = mirror_it->second;
    return true;
}

void ggml_cuda_kv_stream_resident_cache_mark_mirrored(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target) {
    if (cache == nullptr || target == nullptr) {
        return;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end()) {
        return;
    }
    const uint32_t layer = layer_it->second;
    for (const int64_t row : cache->dirty_rows) {
        if (row < 0) {
            return;
        }
        const uint32_t page = uint32_t(uint64_t(row)/cache->page_tokens);
        if (page < cache->layer_pages[layer]) {
            cache->dirty[kv_stream_resident_index(cache, layer, page)] = 0;
        }
    }
}

void ggml_cuda_kv_stream_resident_cache_mark_dirty(
        ggml_cuda_kv_stream_resident_cache * cache,
        const ggml_tensor * target, const ggml_tensor * indices) {
    if (cache == nullptr || target == nullptr || indices == nullptr) {
        return;
    }
    const auto layer_it = cache->layer_by_data.find(target->data);
    if (layer_it == cache->layer_by_data.end()) {
        return;
    }
    const uint32_t layer = layer_it->second;
    GGML_ASSERT(layer < cache->layer_count);
    if (indices->buffer == nullptr || !ggml_backend_buffer_is_host(indices->buffer) ||
            !ggml_is_contiguous(indices) ||
            (indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64)) {
        return;
    }

    cache->precise_dirty_tracking[layer] = 1;
    const size_t begin = cache->layer_offsets[layer];
    const size_t end = cache->layer_offsets[layer + 1];
    const size_t count = ggml_nelements(indices);
    for (size_t i = 0; i < count; ++i) {
        const int64_t row = indices->type == GGML_TYPE_I32 ?
            static_cast<const int32_t *>(indices->data)[i] :
            static_cast<const int64_t *>(indices->data)[i];
        if (row < 0) {
            std::fill(cache->dirty.begin() + begin, cache->dirty.begin() + end, 1);
            return;
        }
        const uint64_t page = uint64_t(row)/cache->page_tokens;
        if (page < cache->layer_pages[layer]) {
            cache->dirty[begin + page] = 1;
        }
    }
}

static uint32_t kv_stream_resident_layer(
        ggml_cuda_kv_stream_resident_cache * cache, const void * k_key) {
    GGML_ASSERT(cache != nullptr);
    auto [it, inserted] = cache->layer_by_k.emplace(k_key, cache->next_layer);
    if (inserted) {
        GGML_ASSERT(cache->next_layer < cache->layer_count);
        ++cache->next_layer;
    }
    return it->second;
}

static bool kv_stream_page_mutable(
        const ggml_cuda_kv_stream_resident_cache * cache,
        uint32_t page) {
    return cache->all_pages_mutable ||
        std::find(cache->mutable_pages.begin(), cache->mutable_pages.end(), page) !=
            cache->mutable_pages.end();
}

namespace {

constexpr int KV_STREAM_HEAD_DIM = 256;
constexpr int KV_STREAM_MAX_PARTS_PER_CHUNK = 16;
constexpr int KV_STREAM_QUERY_WORKSPACE_TOKENS = 256;

static int kv_stream_parts_per_chunk() {
    static const int parts = []() {
        const char * value = getenv("GGML_CUDA_KV_STREAM_PARTS");
        const int parsed = value == nullptr ? 16 : atoi(value);
        return parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16 ? parsed : 16;
    }();
    return parts;
}

// Opt-in only.
//
// The span is a fork optimization that was unreachable while the direct route was dead. It was first
// measured wrong (23 numerical failures, 2e-3 up to 3.5e-2, growing with the KV span) because the MMA
// partial epilogue published the tile_Q meta slot, which the np > 1 combine step overwrites with the
// per-warp *scale* exp(warp_max - combined_max) instead of the per-column (max, rowsum) that
// kv_stream_accumulate_chunk_results needs to weight each chunk. Every weight then evaluated to ~1
// and the chunks merged unweighted, losing the difference between chunk maxima.
//
// Fixing the max alone was not enough: publishing the per-thread KQ_cmr instead only repaired the max
// and left the rowsum a single warp's partial (np warps split the KV range per column), which roughly
// halved it and made the error worse. The correct value is the *combined* (max, rowsum) the np > 1
// combine already computes. The epilogue now publishes (slot[2], slot[1]) where slot[1] holds the
// combined rowsum (KQ_crs) and the combine stashes the combined max (KQ_cmn) in the next padding float
// of the same Q column - slot[0] stays the per-warp scale the numerator combination consumes. For
// np == 1 the slot already holds the complete per-column (max, rowsum).
//
// Set GGML_CUDA_KV_STREAM_MMA_PREFILL=1 to exercise it. test-kv-stream-cuda-attn prints which way a
// run resolved, so an A/B cannot be misread. Note stats.mma_prefill_attention_spans is only
// incremented when the span has a resident cache, so it reads 0 for a runtime built without one.
static bool kv_stream_mma_prefill_enabled() {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CUDA_KV_STREAM_MMA_PREFILL");
        return value != nullptr && atoi(value) != 0;
    }();
    return enabled;
}

// Diagnostic switch for the chunk merge: prints the accumulated (max, denominator) and the first
// four numerator values of the first two rows before they are normalized, so a span-enabled run can
// be compared against a span-disabled (vector) run on the same workload.
static bool kv_stream_span_trace_enabled() {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CUDA_KV_STREAM_TRACE_SPAN");
        return value != nullptr && atoi(value) != 0;
    }();
    return enabled;
}

static int64_t kv_stream_block_tokens(const ggml_tensor * dst, size_t stage_bytes) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
    const size_t bytes_per_token = k_row_bytes*K->ne[2] + v_row_bytes*V->ne[2];
    if (bytes_per_token == 0) {
        return 0;
    }

    int64_t tokens = std::min<int64_t>(K->ne[1], stage_bytes/bytes_per_token);
    tokens = tokens/FATTN_KQ_STRIDE*FATTN_KQ_STRIDE;
    while (tokens > 0) {
        const size_t k_bytes = k_row_bytes*tokens*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_bytes = v_row_bytes*tokens*V->ne[2];
        if (v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset) {
            return tokens;
        }
        tokens -= FATTN_KQ_STRIDE;
    }
    return 0;
}

template<int D>
static __global__ void kv_stream_accumulate_chunk_results(
        const float * parts,
        const float2 * meta,
        float * accumulator,
        float2 * accumulator_meta,
        int nrows,
        bool initialize,
        int nparts) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }
    ggml_cuda_pdl_sync();

    __shared__ float old_maximum;
    __shared__ float maximum;
    __shared__ float old_scale;
    __shared__ float denominator;

    const int base = row*nparts;
    if (tid == 0) {
        old_maximum = initialize ? -FLT_MAX : accumulator_meta[row].x;
        maximum = old_maximum;
        for (int part = 0; part < nparts; ++part) {
            maximum = fmaxf(maximum, meta[base + part].x);
        }

        old_scale = initialize ? 0.0f : expf(old_maximum - maximum);
        denominator = initialize ? 0.0f : old_scale*accumulator_meta[row].y;
        for (int part = 0; part < nparts; ++part) {
            const float weight = expf(meta[base + part].x - maximum);
            denominator += weight*meta[base + part].y;
        }
        accumulator_meta[row] = make_float2(maximum, denominator);
    }
    __syncthreads();

    float numerator = initialize ? 0.0f : old_scale*accumulator[row*D + tid];
    for (int part = 0; part < nparts; ++part) {
        const float weight = expf(meta[base + part].x - maximum);
        numerator += weight*parts[(base + part)*D + tid];
    }
    accumulator[row*D + tid] = numerator;
}

template<int D>
static __global__ void kv_stream_normalize_chunk_results(
        const float * accumulator,
        const float2 * accumulator_meta,
        float * dst,
        int nrows) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }
    ggml_cuda_pdl_sync();

    dst[row*D + tid] = accumulator[row*D + tid]/accumulator_meta[row].y;
}

using kv_stream_native_partial_fn = void (*)(
    ggml_backend_cuda_context &, ggml_tensor *, float *, float2 *, int);

// Partial (per-KV-chunk) kernel of the exact (K,V) pair, D == KV_STREAM_HEAD_DIM, or nullptr.
//
// Whether a pair has that template instance is a per-pair property of the build, not a global
// one: ggml_cuda_fattn_vec_instances() (ggml/cmake/common.cmake) emits one
// GGML_CUDA_FA_<type_K>_<type_V> define per K/V type combination from its FA_TYPES list, and
// only the pairs in FA_COMBINATIONS get their template-instances/fattn-vec-instance-*.cu file
// compiled. `if constexpr` on those same defines is therefore the exact question "was this pair
// compiled", and it is how upstream's own ggml_cuda_get_fattn_vec_case() selects a kernel: an
// uncompiled pair degrades to a null case there and to nullptr here, instead of referencing a
// symbol the linker does not have.
//
// F32 is deliberately absent: it has no partial template instance for any partner (upstream's
// generic selector maps F32 onto the F16 case instead, which the streamed native path does not
// do - it stages whatever the KV cache holds).
#define KV_STREAM_NATIVE_PARTIAL_CASE(type_K_case, type_V_case)                        \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                        \
        if (type_k == GGML_TYPE_##type_K_case && type_v == GGML_TYPE_##type_V_case) {  \
            return ggml_cuda_flash_attn_ext_vec_partial_case<                          \
                KV_STREAM_HEAD_DIM, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>; \
        }                                                                             \
    }

// Same K/V pair set as ggml_cuda_get_fattn_vec_case() below, TurboQuant first because that is
// what the streamed KV cache is normally asked for.
static kv_stream_native_partial_fn kv_stream_resolve_native_partial(
        ggml_type type_k, ggml_type type_v) {
    // TurboQuant KV-cache types.
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_0, TURBO2_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_0, TURBO3_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0, TURBO4_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,      TURBO2_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,     TURBO2_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_0, Q8_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,      TURBO3_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,     TURBO3_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_0, Q8_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,      TURBO4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,     TURBO4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0, Q8_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_0, TURBO3_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_0, TURBO2_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_0, TURBO4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0, TURBO2_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_0, TURBO4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0, TURBO3_0)

    // TurboQuant turbo8 / TCQ (ported from buun-llama-cpp): same treatment as turbo2/3/4 above.
    // DECL_FATTN_VEC_CASE already instantiates the streamed partial case alongside the regular
    // one for every pair that has an instance file, so no extra kernel work is needed here --
    // the native streamed attention path becomes available for these types too. Pairs whose
    // instance was not compiled are skipped by the `if constexpr (GGML_CUDA_FA_*)` gate, which
    // keeps this list in sync with what ggml_cuda_flash_attn_ext_vec_partial_case can find.
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, TURBO8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(F16,      TURBO8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,     TURBO8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, TURBO4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0, TURBO8_0)

    // Mixed turbo8 / standard-quant / TCQ pairs (same set as the vec case table above):
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO8_0, Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1,     TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1,     TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1,     TURBO4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1,     TURBO4_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_TCQ, TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_TCQ, TURBO2_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_TCQ, TURBO2_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_TCQ, TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(F16,        TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(F16,        TURBO2_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_TCQ, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_TCQ, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,       TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,       TURBO2_TCQ)

    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO1_TCQ, TURBO1_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO3_TCQ, TURBO1_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO2_TCQ, TURBO1_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO1_TCQ, TURBO3_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO1_TCQ, TURBO2_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(F16,        TURBO1_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO1_TCQ, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0,       TURBO1_TCQ)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO1_TCQ, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(TURBO4_0,   TURBO1_TCQ)

    // Base K/V types.
    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, F16)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, F16)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  Q4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, Q4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, Q4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, Q4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, Q4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, Q4_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, Q4_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, Q4_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, Q4_1)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  Q5_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, Q5_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, Q5_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, Q5_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, Q5_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, Q5_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, Q5_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, Q5_1)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, Q5_1)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, Q8_0)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, Q8_0)

    KV_STREAM_NATIVE_PARTIAL_CASE(F16,  BF16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_0, BF16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q4_1, BF16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_0, BF16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q5_1, BF16)
    KV_STREAM_NATIVE_PARTIAL_CASE(Q8_0, BF16)
    KV_STREAM_NATIVE_PARTIAL_CASE(BF16, BF16)

    return nullptr;
}
#undef KV_STREAM_NATIVE_PARTIAL_CASE

} // namespace

struct ggml_backend_cuda_kv_stream_type_capabilities
ggml_backend_cuda_kv_stream_get_type_capabilities(ggml_type type) {
    ggml_backend_cuda_kv_stream_type_capabilities result{};

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
            result.classified = true;
            break;
        default:
            return result;
    }

    result.storage = type != GGML_TYPE_Q8_1 && type != GGML_TYPE_Q8_K;

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        // TurboQuant: k_set_rows_turbo* quantise the rows on the device, including the
        // forward WHT and the InnerQ calibration, so the cache can be written online.
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        // turbo8 / TCQ use k_set_rows_turbo8 / k_set_rows_turbo_tcq (Viterbi) on device too.
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
            result.online_write = true;
            break;
        default:
            break;
    }

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        // TurboQuant: ggml_get_to_fp16_cuda dequantises the turbo blocks element-wise, which
        // is exactly the transform the streamed pages need before the F16 fallback attention.
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
            result.decode_f16 = true;
            break;
        default:
            break;
    }

    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        // TurboQuant: the streamed native path consumes the rotated cache blocks directly, via
        // the same ggml_cuda_flash_attn_ext_vec_partial_case <K,V> instance that non-streamed
        // turbo attention uses. The forward WHT on Q and the inverse rotation on the attention
        // output are graph ops (llama-graph.cpp), so they are unaffected by which FA kernel runs.
        // turbo8 / TCQ are included because DECL_FATTN_VEC_CASE instantiates their streamed
        // partial case as well; kv_stream_resolve_native_partial() below lists the valid pairs.
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
            result.direct_attention = true;
            break;
        default:
            break;
    }

    result.requires_initialization =
        type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S;
    result.requires_importance_matrix =
        type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M;
    result.auxiliary = type == GGML_TYPE_Q8_1 || type == GGML_TYPE_Q8_K;
    return result;
}

ggml_backend_cuda_kv_stream_attention_mode
ggml_backend_cuda_kv_stream_get_attention_mode(ggml_type type_k, ggml_type type_v) {
    const auto capabilities_k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
    const auto capabilities_v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
    // DIRECT is a property of the (K,V) pair, not of either type: direct_attention only says the
    // type has a native streamed kernel for some partner. Resolve the pair and only then report
    // DIRECT, so the mode can never promise a kernel that ggml_cuda_flash_attn_ext_streamed()
    // would fail to find - that mismatch used to be a GGML_ASSERT(native_partial != nullptr),
    // i.e. an abort on the first streamed layer. Pairs whose instance was not compiled now fall
    // through to the F16 conversion path below, which is exactly what upstream's generic
    // selector does for them.
    if (capabilities_k.direct_attention && capabilities_v.direct_attention &&
            kv_stream_resolve_native_partial(type_k, type_v) != nullptr) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT;
    }
    if (capabilities_k.storage && capabilities_v.storage &&
            capabilities_k.online_write && capabilities_v.online_write &&
            capabilities_k.decode_f16 && capabilities_v.decode_f16 &&
            !capabilities_k.requires_importance_matrix && !capabilities_v.requires_importance_matrix) {
        return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
    }
    return GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED;
}

bool ggml_cuda_kv_stream_page_bytes(
        ggml_type type_k, ggml_type type_v,
        uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
        uint32_t page_tokens, size_t * page_bytes) {
    if (head_dim_k == 0 || head_dim_v == 0 || head_count == 0 || page_tokens == 0 || page_bytes == nullptr) {
        return false;
    }

    const auto capabilities_k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
    const auto capabilities_v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
    if (!capabilities_k.storage || !capabilities_v.storage) {
        return false;
    }

    const uint32_t block_size_k = ggml_blck_size(type_k);
    const uint32_t block_size_v = ggml_blck_size(type_v);
    if (block_size_k == 0 || block_size_v == 0 ||
            head_dim_k % block_size_k != 0 || head_dim_v % block_size_v != 0) {
        return false;
    }

    const size_t maximum = std::numeric_limits<size_t>::max();
    const size_t row_bytes_k = ggml_row_size(type_k, head_dim_k);
    const size_t row_bytes_v = ggml_row_size(type_v, head_dim_v);
    if (row_bytes_k > maximum/head_count || row_bytes_v > maximum/head_count) {
        return false;
    }
    const size_t token_bytes_k = row_bytes_k*head_count;
    const size_t token_bytes_v = row_bytes_v*head_count;
    if (token_bytes_k > maximum/page_tokens || token_bytes_v > maximum/page_tokens) {
        return false;
    }
    const size_t page_bytes_k = token_bytes_k*page_tokens;
    const size_t page_bytes_v = token_bytes_v*page_tokens;
    if (page_bytes_k > maximum - 127) {
        return false;
    }
    const size_t page_offset_v = (page_bytes_k + 127) & ~size_t(127);
    if (page_bytes_v > maximum - page_offset_v) {
        return false;
    }

    *page_bytes = page_offset_v + page_bytes_v;
    return true;
}

bool ggml_cuda_kv_stream_workspace_bytes(
        ggml_type type_k, ggml_type type_v,
        uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
        uint32_t page_tokens, size_t * workspace_bytes) {
    if (workspace_bytes == nullptr) {
        return false;
    }
    const auto mode =
        ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v);
    if (mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT) {
        *workspace_bytes = 0;
        return true;
    }
    if (mode != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16) {
        return false;
    }
    return ggml_cuda_kv_stream_page_bytes(
        GGML_TYPE_F16, GGML_TYPE_F16,
        head_dim_k, head_dim_v, head_count, page_tokens, workspace_bytes);
}

const char * ggml_cuda_flash_attn_ext_streamed_reject_reason(const ggml_tensor * dst, size_t stage_bytes) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return "node is not a FLASH_ATTN_EXT";
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return "missing Q, K or V operand";
    }
    if (Q->type != GGML_TYPE_F32) {
        return "Q is not F32";
    }
    if (ggml_backend_cuda_kv_stream_get_attention_mode(K->type, V->type) ==
            GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED) {
        return "this (K,V) type pair has no streamed attention mode";
    }
    if (Q->ne[0] != KV_STREAM_HEAD_DIM) {
        return "Q head dim is not KV_STREAM_HEAD_DIM (256)";
    }
    if (V->ne[0] != KV_STREAM_HEAD_DIM) {
        return "V head dim is not KV_STREAM_HEAD_DIM (256)";
    }
    if (Q->ne[1] < 1) {
        return "Q has no queries";
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return "batch dim (ne[3]) is not 1 for Q, K or V";
    }
    if (K->ne[1] != V->ne[1]) {
        return "K and V have different token counts";
    }
    if (K->ne[2] != V->ne[2]) {
        return "K and V have different head counts";
    }
    if (K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return "KV token count is not a multiple of FATTN_KQ_STRIDE (256)";
    }
    if (K->nb[0] != ggml_element_size(K) || V->nb[0] != ggml_element_size(V)) {
        return "K or V nb[0] is not the element size (not tightly packed)";
    }
    if (K->nb[1] < ggml_row_size(K->type, K->ne[0]) ||
        V->nb[1] < ggml_row_size(V->type, V->ne[0])) {
        return "K or V row stride is smaller than one row";
    }
    if (K->nb[1] != ggml_row_size(K->type, K->ne[0])*K->ne[2]) {
        return "K nb[1] != row_size * head_count (not the streamed page layout)";
    }
    if (K->nb[2] != ggml_row_size(K->type, K->ne[0])) {
        return "K nb[2] != one row (not the streamed page layout)";
    }
    if (V->nb[1] != ggml_row_size(V->type, V->ne[0])*V->ne[2]) {
        return "V nb[1] != row_size * head_count (not the streamed page layout)";
    }
    if (V->nb[2] != ggml_row_size(V->type, V->ne[0])) {
        return "V nb[2] != one row (not the streamed page layout)";
    }
    if (mask != nullptr && !(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask))) {
        return "mask is not a contiguous F16 tensor";
    }
    if (sinks != nullptr) {
        return "attention sinks are not supported by the streamed path";
    }
    if (kv_stream_block_tokens(dst, stage_bytes) <= 0) {
        return "stage is too small to hold one FATTN_KQ_STRIDE block of K+V";
    }
    return nullptr;
}

bool ggml_cuda_flash_attn_ext_streamed_supported(const ggml_tensor * dst, size_t stage_bytes) {
    return ggml_cuda_flash_attn_ext_streamed_reject_reason(dst, stage_bytes) == nullptr;
}

namespace {

static __global__ void kv_stream_record_deadline(
        const uint32_t * ready_flag,
        uint64_t * samples,
        uint64_t * misses) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long *>(samples), 1ULL);
        if (*ready_flag == 0) {
            atomicAdd(reinterpret_cast<unsigned long long *>(misses), 1ULL);
        }
    }
}

static bool kv_stream_collect_timing(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (!ring->timing_pending) {
        return true;
    }
    const cudaError_t status = cudaEventQuery(ring->eval_end);
    if (status == cudaErrorNotReady) {
        return false;
    }
    CUDA_CHECK(status);

    float eval_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eval_ms, ring->eval_start, ring->eval_end));
    double busy_ratio = 0.0;
    if (ring->copy_sample_recorded && eval_ms > 0.0f && ring->current_epoch_uploads > 0) {
        float copy_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &copy_ms, ring->copy_sample_start, ring->copy_sample_end));
        GGML_ASSERT(ring->copy_sample_uploads > 0);
        busy_ratio = std::min(1.0, double(copy_ms)*double(ring->current_epoch_uploads)/
            (double(ring->copy_sample_uploads)*double(eval_ms)));
    }
    ring->last_copy_engine_busy_ratio = busy_ratio;
    ring->last_ring_peak_occupancy = ring->current_ring_peak_occupancy;
    ring->timing_pending = false;
    return true;
}

static void kv_stream_graph_upload(
        ggml_cuda_kv_stream_transfer_ring * ring,
        kv_stream_graph_request & request,
        uint32_t slot) {
    // Preserve the host cache's compact token-major layout. This makes each
    // page one contiguous K transfer plus one contiguous V transfer.
    char * k_stage = ring->pool_data + size_t(slot)*request.k_bytes;
    char * v_stage = ring->pool_data +
        size_t(ring->active_slots)*request.k_bytes + size_t(slot)*request.v_bytes;
    if (ring->slot_used[slot]) {
        CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
        ++ring->stage_slot_reuses;
    }
    if (request.deadline_sample) {
        CUDA_CHECK(cudaMemsetAsync(
            ring->ready_flags_device + slot, 0, sizeof(uint32_t), ring->copy_stream));
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_start, ring->copy_stream));
    }
    GGML_ASSERT(request.k_nb1 == request.k_row_bytes*request.n_head_kv);
    GGML_ASSERT(request.k_nb2 == request.k_row_bytes);
    GGML_ASSERT(request.v_nb1 == request.v_row_bytes*request.n_head_kv);
    GGML_ASSERT(request.v_nb2 == request.v_row_bytes);
    CUDA_CHECK(cudaMemcpyAsync(
        k_stage, request.k_data + request.token_begin*request.k_nb1,
        request.k_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_stage, request.v_data + request.token_begin*request.v_nb1,
        request.v_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    ring->host_to_device_copy_commands += 2;
    if (request.deadline_sample) {
        CUDA_CHECK(cudaMemsetAsync(
            ring->ready_flags_device + slot, 1, sizeof(uint32_t), ring->copy_stream));
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_end, ring->copy_stream));
        ring->copy_sample_recorded = true;
        ring->copy_sample_uploads = 1;
    }
    CUDA_CHECK(cudaEventRecord(ring->ready[slot], ring->copy_stream));
    request.slot = slot;
    request.scheduled = true;
    ring->slot_used[slot] = 1;
    request.ready_slot = slot;
    ring->slot_request[slot] = size_t(&request - ring->graph_requests.data());
    ++ring->current_occupancy;
    ring->ring_peak_occupancy = std::max(
        ring->ring_peak_occupancy, ring->current_occupancy);
    ring->current_ring_peak_occupancy = std::max(
        ring->current_ring_peak_occupancy, ring->current_occupancy);
    ++ring->asynchronous_page_uploads;
    ++ring->current_epoch_uploads;
    if (ring->graph_resident_cache != nullptr) {
        ring->graph_resident_cache->stats.host_to_device_bytes +=
            request.k_bytes + request.v_bytes;
    }
    if (ring->current_layer != KV_STREAM_NO_LAYER && request.layer > ring->current_layer) {
        ++ring->cross_layer_prefetches;
    }
}

static bool kv_stream_graph_request_follows(
        const kv_stream_graph_request & previous,
        const kv_stream_graph_request & request) {
    return request.eligible && !request.scheduled && !request.consumed &&
        request.layer == previous.layer &&
        request.k_data == previous.k_data && request.v_data == previous.v_data &&
        request.k_nb1 == previous.k_nb1 && request.v_nb1 == previous.v_nb1 &&
        request.k_bytes == previous.k_bytes && request.v_bytes == previous.v_bytes &&
        request.token_count == previous.token_count &&
        request.token_begin == previous.token_begin + previous.token_count;
}

static void kv_stream_graph_upload_batch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        size_t first_request,
        uint32_t first_slot,
        uint32_t batch_pages) {
    GGML_ASSERT(batch_pages > 1);
    GGML_ASSERT(first_request + batch_pages <= ring->graph_requests.size());
    GGML_ASSERT(first_slot + batch_pages <= ring->active_slots);
    auto & first = ring->graph_requests[first_request];

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        const uint32_t slot = first_slot + page;
        GGML_ASSERT(request.eligible && !request.scheduled && !request.consumed);
        GGML_ASSERT(ring->slot_request[slot] == KV_STREAM_NO_REQUEST);
        if (ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
            ++ring->stage_slot_reuses;
        }
        if (request.deadline_sample) {
            CUDA_CHECK(cudaMemsetAsync(
                ring->ready_flags_device + slot, 0, sizeof(uint32_t), ring->copy_stream));
        }
    }

    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_start, ring->copy_stream));
    }
    GGML_ASSERT(first.k_nb1 == first.k_row_bytes*first.n_head_kv);
    GGML_ASSERT(first.k_nb2 == first.k_row_bytes);
    GGML_ASSERT(first.v_nb1 == first.v_row_bytes*first.n_head_kv);
    GGML_ASSERT(first.v_nb2 == first.v_row_bytes);
    char * k_stage = ring->pool_data + size_t(first_slot)*first.k_bytes;
    char * v_stage = ring->pool_data +
        size_t(ring->active_slots)*first.k_bytes + size_t(first_slot)*first.v_bytes;
    CUDA_CHECK(cudaMemcpyAsync(
        k_stage, first.k_data + first.token_begin*first.k_nb1,
        size_t(batch_pages)*first.k_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_stage, first.v_data + first.token_begin*first.v_nb1,
        size_t(batch_pages)*first.v_bytes, cudaMemcpyHostToDevice, ring->copy_stream));
    ring->host_to_device_copy_commands += 2;

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        if (request.deadline_sample) {
            CUDA_CHECK(cudaMemsetAsync(
                ring->ready_flags_device + first_slot + page, 1,
                sizeof(uint32_t), ring->copy_stream));
        }
    }
    if (ring->timing_current && !ring->copy_sample_recorded) {
        CUDA_CHECK(cudaEventRecord(ring->copy_sample_end, ring->copy_stream));
        ring->copy_sample_recorded = true;
        ring->copy_sample_uploads = batch_pages;
    }
    CUDA_CHECK(cudaEventRecord(ring->ready[first_slot], ring->copy_stream));

    for (uint32_t page = 0; page < batch_pages; ++page) {
        auto & request = ring->graph_requests[first_request + page];
        const uint32_t slot = first_slot + page;
        request.slot = slot;
        request.ready_slot = first_slot;
        request.scheduled = true;
        ring->slot_used[slot] = 1;
        ring->slot_request[slot] = first_request + page;
        ++ring->current_occupancy;
        ++ring->asynchronous_page_uploads;
        ++ring->current_epoch_uploads;
        if (ring->graph_resident_cache != nullptr) {
            ring->graph_resident_cache->stats.host_to_device_bytes +=
                request.k_bytes + request.v_bytes;
        }
        if (ring->current_layer != KV_STREAM_NO_LAYER && request.layer > ring->current_layer) {
            ++ring->cross_layer_prefetches;
        }
    }
    ring->ring_peak_occupancy = std::max(
        ring->ring_peak_occupancy, ring->current_occupancy);
    ring->current_ring_peak_occupancy = std::max(
        ring->current_ring_peak_occupancy, ring->current_occupancy);
}

static uint32_t kv_stream_graph_batch_pages(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t first_slot) {
    if (!ring->graph_active || first_slot >= ring->active_slots ||
            ring->next_request >= ring->graph_requests.size() ||
            ring->slot_request[first_slot] != KV_STREAM_NO_REQUEST) {
        return 0;
    }
    const auto & first = ring->graph_requests[ring->next_request];
    if (!first.eligible) {
        return 0;
    }
    GGML_ASSERT(!first.scheduled && !first.consumed);

    const uint32_t maximum = std::min<uint32_t>({
        KV_STREAM_COPY_BATCH_PAGES,
        ring->active_slots - first_slot,
        uint32_t(ring->graph_requests.size() - ring->next_request),
    });
    uint32_t pages = 1;
    while (pages < maximum) {
        if (ring->slot_request[first_slot + pages] != KV_STREAM_NO_REQUEST ||
                !kv_stream_graph_request_follows(
                    ring->graph_requests[ring->next_request + pages - 1],
                    ring->graph_requests[ring->next_request + pages])) {
            break;
        }
        ++pages;
    }
    return pages;
}

static uint32_t kv_stream_graph_schedule_batch(
        ggml_cuda_kv_stream_transfer_ring * ring,
        uint32_t first_slot) {
    const uint32_t batch_pages = kv_stream_graph_batch_pages(ring, first_slot);
    if (batch_pages == 0) {
        return 0;
    }
    const size_t first_request = ring->next_request;
    // Probe every immutable copy batch at its actual compute deadline. Mutable
    // tails are produced by this graph and are excluded from prefetch quality
    // feedback so they do not force unnecessary resident-page demotions.
    ring->graph_requests[first_request].deadline_sample =
        !ring->graph_requests[first_request].mutable_tail;
    ring->next_request += batch_pages;
    if (batch_pages == 1) {
        kv_stream_graph_upload(
            ring, ring->graph_requests[first_request], first_slot);
    } else {
        kv_stream_graph_upload_batch(
            ring, first_request, first_slot, batch_pages);
    }
    return batch_pages;
}

static void kv_stream_graph_fill_free_slots(ggml_cuda_kv_stream_transfer_ring * ring) {
    for (uint32_t slot = 0; slot < ring->active_slots; ++slot) {
        if (ring->slot_request[slot] != KV_STREAM_NO_REQUEST) {
            continue;
        }
        const uint32_t scheduled = kv_stream_graph_schedule_batch(ring, slot);
        if (scheduled == 0) {
            break;
        }
        slot += scheduled - 1;
    }
}

static bool kv_stream_graph_layer_begin(
        ggml_cuda_kv_stream_transfer_ring * ring,
        const void * k_key,
        cudaStream_t compute_stream) {
    if (!ring->graph_active) {
        return false;
    }
    const auto layer_it = ring->graph_layer_by_k.find(k_key);
    if (layer_it == ring->graph_layer_by_k.end()) {
        return false;
    }

    ring->current_layer = layer_it->second;
    CUDA_CHECK(cudaEventRecord(ring->producer_ready, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->producer_ready, 0));
    for (auto & request : ring->graph_requests) {
        if (request.layer == ring->current_layer && request.mutable_tail) {
            request.eligible = true;
        }
    }
    kv_stream_graph_fill_free_slots(ring);
    return true;
}

static size_t kv_stream_graph_request_index(
        const ggml_cuda_kv_stream_transfer_ring * ring,
        const void * k_key,
        uint32_t page) {
    const auto it = ring->graph_request_by_k_page.find(k_key);
    if (it == ring->graph_request_by_k_page.end() || page >= it->second.size()) {
        return KV_STREAM_NO_REQUEST;
    }
    return it->second[page];
}

static void kv_stream_graph_release(
        ggml_cuda_kv_stream_transfer_ring * ring,
        size_t request_index,
        cudaStream_t compute_stream) {
    GGML_ASSERT(request_index < ring->graph_requests.size());
    auto & request = ring->graph_requests[request_index];
    GGML_ASSERT(request.scheduled && !request.consumed);
    const uint32_t slot = request.slot;
    GGML_ASSERT(ring->slot_request[slot] == request_index);
    CUDA_CHECK(cudaEventRecord(ring->consumed[slot], compute_stream));
    request.consumed = true;
    ring->slot_request[slot] = KV_STREAM_NO_REQUEST;
    GGML_ASSERT(ring->current_occupancy > 0);
    --ring->current_occupancy;
}

} // namespace

void ggml_cuda_kv_stream_graph_begin(ggml_cuda_kv_stream_transfer_ring * ring) {
    GGML_ASSERT(ring != nullptr);
    const bool timing_available = kv_stream_collect_timing(ring);
    ring->graph_active = true;
    ring->graph_decode = true;
    ring->graph_decode_span_pages = ring->forced_decode_span_pages != 0 ?
        ring->forced_decode_span_pages :
        (ring->span_tuner.use_bounded() ? KV_STREAM_DECODE_SPAN_PAGES : UINT32_MAX);
    ring->graph_layer_count = 0;
    ring->current_layer = KV_STREAM_NO_LAYER;
    ring->next_request = 0;
    ring->graph_resident_cache = nullptr;
    ring->graph_requests.clear();
    ring->graph_layer_by_k.clear();
    ring->graph_request_by_k_page.clear();
    ring->current_occupancy = 0;
    if (timing_available) {
        ring->current_ring_peak_occupancy = 0;
        ring->current_epoch_uploads = 0;
        ring->copy_sample_recorded = false;
        ring->copy_sample_uploads = 0;
    }
    ring->timing_current = timing_available;
    std::fill(ring->slot_request.begin(), ring->slot_request.end(), KV_STREAM_NO_REQUEST);
}

bool ggml_cuda_kv_stream_graph_add_attention(
        ggml_cuda_kv_stream_transfer_ring * ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache,
        const ggml_tensor * dst) {
    GGML_ASSERT(ring != nullptr);
    const ggml_tensor * K = dst == nullptr ? nullptr : dst->src[1];
    const ggml_tensor * V = dst == nullptr ? nullptr : dst->src[2];
    if (resident_cache == nullptr || dst == nullptr ||
            !ggml_cuda_flash_attn_ext_streamed_supported(dst, ring->page_bytes)) {
        return false;
    }
    if (ring->graph_resident_cache != nullptr && ring->graph_resident_cache != resident_cache) {
        return false;
    }
    ring->graph_decode = ring->graph_decode && dst->src[0]->ne[1] == 1;
    if (dst->src[0]->ne[1] != 1) {
        // Graphs are rebuilt across warmup, prompt chunks, and slot reuse.
        // Relearn pointer-to-layer identity once per prefill graph while the
        // resident page contents are refreshed by the local multi-token path.
        if (ring->graph_resident_cache == nullptr) {
            resident_cache->layer_by_k.clear();
            resident_cache->layer_by_data.clear();
            resident_cache->next_layer = 0;
            ring->graph_resident_cache = resident_cache;
        }
        const uint32_t resident_layer = kv_stream_resident_layer(resident_cache, K->data);
        resident_cache->layer_by_data[K->data] = resident_layer;
        resident_cache->layer_by_data[V->data] = resident_layer;
        char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
            resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
        resident_cache->mirror_by_data[K->data] = layer_base;
        resident_cache->mirror_by_data[V->data] = layer_base +
            size_t(resident_cache->layer_pages[resident_layer])*K->nb[1]*resident_cache->page_tokens;
        return false;
    }
    ring->graph_resident_cache = resident_cache;
    // Resident placement is stable across evaluations, while deadlines must
    // follow this graph's finalized execution order.
    const uint32_t resident_layer = kv_stream_resident_layer(resident_cache, K->data);
    resident_cache->layer_by_data[K->data] = resident_layer;
    resident_cache->layer_by_data[V->data] = resident_layer;
    char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
        resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
    resident_cache->mirror_by_data[K->data] = layer_base;
    resident_cache->mirror_by_data[V->data] = layer_base +
        size_t(resident_cache->layer_pages[resident_layer])*K->nb[1]*resident_cache->page_tokens;

    const int64_t block_tokens = resident_cache->page_tokens;
    const int nchunks = int((K->ne[1] + block_tokens - 1)/block_tokens);
    if (uint32_t(nchunks) <= resident_cache->layer_pages[resident_layer]) {
        return true;
    }
    const uint32_t layer = ring->graph_layer_count++;
    ring->graph_layer_by_k[K->data] = layer;
    auto & page_requests = ring->graph_request_by_k_page[K->data];
    page_requests.assign(nchunks, KV_STREAM_NO_REQUEST);

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const uint32_t page = uint32_t(chunk);
        if (page < resident_cache->layer_pages[resident_layer]) {
            continue;
        }
        const int64_t token_begin = chunk*block_tokens;
        const int64_t token_count = std::min<int64_t>(block_tokens, K->ne[1] - token_begin);
        const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
        const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);

        kv_stream_graph_request request;
        request.k_data = static_cast<const char *>(K->data);
        request.v_data = static_cast<const char *>(V->data);
        request.k_nb1 = K->nb[1];
        request.k_nb2 = K->nb[2];
        request.v_nb1 = V->nb[1];
        request.v_nb2 = V->nb[2];
        request.n_head_kv = K->ne[2];
        request.token_begin = token_begin;
        request.token_count = token_count;
        request.k_row_bytes = k_row_bytes;
        request.v_row_bytes = v_row_bytes;
        request.k_head_bytes = k_row_bytes*token_count;
        request.k_bytes = request.k_head_bytes*K->ne[2];
        request.v_offset = GGML_PAD(request.k_bytes, 128);
        request.v_head_bytes = v_row_bytes*token_count;
        request.v_bytes = request.v_head_bytes*V->ne[2];
        request.layer = layer;
        request.mutable_tail = chunk == nchunks - 1 ||
            kv_stream_page_mutable(resident_cache, page);
        request.eligible = !request.mutable_tail;
        GGML_ASSERT(request.v_offset + request.v_bytes == ring->page_bytes);

        page_requests[page] = ring->graph_requests.size();
        ring->graph_requests.push_back(request);
    }
    return true;
}

void ggml_cuda_kv_stream_graph_finalize(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream) {
    GGML_ASSERT(ring != nullptr);
    ring->last_graph_decode = ring->graph_decode;
    ring->last_graph_bounded = ring->graph_decode_span_pages != UINT32_MAX;
    ring->last_graph_streamed = !ring->graph_requests.empty();
    if (ring->graph_requests.empty()) {
        ring->timing_current = false;
        return;
    }
    if (ring->timing_current) {
        CUDA_CHECK(cudaEventRecord(ring->eval_start, compute_stream));
    }
    kv_stream_graph_fill_free_slots(ring);
}

void ggml_cuda_kv_stream_graph_end(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream) {
    GGML_ASSERT(ring != nullptr);
    if (ring->timing_current) {
        CUDA_CHECK(cudaEventRecord(ring->eval_end, compute_stream));
        ring->timing_pending = true;
        ring->timing_current = false;
    }
}

double ggml_cuda_kv_stream_copy_engine_busy_ratio(
        ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return 0.0;
    }
    (void) kv_stream_collect_timing(ring);
    return ring->last_copy_engine_busy_ratio;
}

uint32_t ggml_cuda_kv_stream_last_ring_peak_occupancy(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    return ring == nullptr ? 0 : ring->last_ring_peak_occupancy;
}

// Defined next to ggml_cuda_flash_attn_ext_get_alloc_size() below; answers whether the KQV
// allocation reserves the f16 scratch that ggml_cuda_flash_attn_ext_get_f16_extra_data() hands out.
static void ggml_cuda_flash_attn_ext_needs_f16(
    const ggml_tensor * dst, bool * need_f16_K, bool * need_f16_V);

void ggml_cuda_flash_attn_ext_streamed(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_kv_stream_transfer_ring * transfer_ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache) {
    GGML_ASSERT(transfer_ring != nullptr);
    void * stage_data = transfer_ring->pool_data;
    const size_t stage_bytes = transfer_ring->page_bytes;
    GGML_ASSERT(ggml_cuda_flash_attn_ext_streamed_supported(dst, stage_bytes));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const auto attention_mode =
        ggml_backend_cuda_kv_stream_get_attention_mode(K->type, V->type);
    const bool convert_to_f16 =
        attention_mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
    // get_attention_mode() only reports DIRECT after resolving this exact pair, so this is a
    // belt-and-braces re-derivation rather than the place where the decision is made.
    const kv_stream_native_partial_fn native_partial =
        attention_mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ?
            kv_stream_resolve_native_partial(K->type, V->type) : nullptr;
    GGML_ASSERT(attention_mode != GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ||
        native_partial != nullptr);
    const to_fp16_cuda_t converter_k = convert_to_f16 && K->type != GGML_TYPE_F16 ?
        ggml_get_to_fp16_cuda(K->type) : nullptr;
    const to_fp16_cuda_t converter_v = convert_to_f16 && V->type != GGML_TYPE_F16 ?
        ggml_get_to_fp16_cuda(V->type) : nullptr;
    GGML_ASSERT(!convert_to_f16 || K->type == GGML_TYPE_F16 || converter_k != nullptr);
    GGML_ASSERT(!convert_to_f16 || V->type == GGML_TYPE_F16 || converter_v != nullptr);
    const int64_t block_tokens = resident_cache != nullptr ?
        resident_cache->page_tokens : kv_stream_block_tokens(dst, stage_bytes);
    const int nchunks = (K->ne[1] + block_tokens - 1)/block_tokens;
    const int nrows = ggml_nrows(dst);
    const uint32_t maximum_streamed_span_pages = Q->ne[1] == 1 ?
        transfer_ring->graph_decode_span_pages : UINT32_MAX;

    struct chunk_descriptor {
        int64_t token_begin = 0;
        int64_t token_count = 0;
        size_t k_row_bytes = 0;
        size_t v_row_bytes = 0;
        size_t k_head_bytes = 0;
        size_t k_bytes = 0;
        size_t v_offset = 0;
        size_t v_head_bytes = 0;
        size_t v_bytes = 0;
        char * k_stage = nullptr;
        char * v_stage = nullptr;
        size_t k_stage_token_stride = 0;
        size_t k_stage_head_stride = 0;
        size_t v_stage_token_stride = 0;
        size_t v_stage_head_stride = 0;
        bool upload = true;
        bool resident_refresh = false;
        bool streamed = false;
        uint32_t slot = 0;
        size_t request_index = KV_STREAM_NO_REQUEST;
    };

    uint32_t resident_layer = 0;
    if (resident_cache != nullptr) {
        resident_layer = kv_stream_resident_layer(resident_cache, K->data);
    }
    const uint32_t resident_layer_pages = resident_cache == nullptr ?
        0 : resident_cache->layer_pages[resident_layer];
    const bool graph_planned = kv_stream_graph_layer_begin(
        transfer_ring, K->data, ctx.stream());

    std::vector<chunk_descriptor> chunks(nchunks);
    std::vector<size_t> streamed_chunks;
    streamed_chunks.reserve(nchunks);

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        const int64_t token_begin = chunk*block_tokens;
        const int64_t token_count = std::min<int64_t>(block_tokens, K->ne[1] - token_begin);
        const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
        const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
        const size_t k_head_bytes = k_row_bytes*token_count;
        const size_t k_bytes = k_head_bytes*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_head_bytes = v_row_bytes*token_count;
        const size_t v_bytes = v_head_bytes*V->ne[2];
        GGML_ASSERT(v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset);

        desc.token_begin = token_begin;
        desc.token_count = token_count;
        desc.k_row_bytes = k_row_bytes;
        desc.v_row_bytes = v_row_bytes;
        desc.k_head_bytes = k_head_bytes;
        desc.k_bytes = k_bytes;
        desc.v_offset = v_offset;
        desc.v_head_bytes = v_head_bytes;
        desc.v_bytes = v_bytes;
        desc.k_stage = static_cast<char *>(stage_data);
        desc.v_stage = desc.k_stage + v_offset;
        desc.k_stage_token_stride = k_row_bytes*K->ne[2];
        desc.k_stage_head_stride = k_row_bytes;
        desc.v_stage_token_stride = v_row_bytes*V->ne[2];
        desc.v_stage_head_stride = v_row_bytes;

        if (resident_cache != nullptr) {
            GGML_ASSERT(token_count == resident_cache->page_tokens);
            GGML_ASSERT(v_offset + v_bytes == resident_cache->page_bytes);

            const uint32_t page = token_begin/resident_cache->page_tokens;
            if (page < resident_layer_pages) {
                const size_t resident_index =
                    kv_stream_resident_index(resident_cache, resident_layer, page);
                // Keep each layer's resident K and V in separate token-major
                // planes so resident pages form one directly consumable span.
                char * layer_base = resident_cache->pool_data + resident_cache->scratch_bytes +
                    resident_cache->layer_offsets[resident_layer]*resident_cache->page_bytes;
                const size_t resident_k_plane_bytes =
                    size_t(resident_layer_pages)*k_bytes;
                desc.k_stage = layer_base + size_t(page)*k_bytes;
                desc.v_stage = layer_base + resident_k_plane_bytes + size_t(page)*v_bytes;
                if (resident_cache->loaded[resident_index]) {
                    ++resident_cache->stats.resident_hits;
                    desc.upload = resident_cache->precise_dirty_tracking[resident_layer] ?
                        resident_cache->dirty[resident_index] :
                        (dst->src[0]->ne[1] > 1 || chunk == nchunks - 1);
                    desc.resident_refresh = desc.upload;
                } else {
                    ++resident_cache->stats.resident_misses;
                    resident_cache->loaded[resident_index] = 1;
                }
            } else {
                ++resident_cache->stats.streamed_pages;
                desc.streamed = true;
            }
        } else {
            desc.streamed = true;
        }

        if (desc.streamed) {
            const size_t stream_index = streamed_chunks.size();
            if (graph_planned) {
                desc.request_index = kv_stream_graph_request_index(
                    transfer_ring, K->data, uint32_t(chunk));
                GGML_ASSERT(desc.request_index != KV_STREAM_NO_REQUEST);
            } else {
                desc.slot = uint32_t(stream_index%transfer_ring->active_slots);
                desc.k_stage = transfer_ring->pool_data + size_t(desc.slot)*desc.k_bytes;
                desc.v_stage = transfer_ring->pool_data +
                    size_t(transfer_ring->active_slots)*desc.k_bytes +
                    size_t(desc.slot)*desc.v_bytes;
            }
            streamed_chunks.push_back(chunk);
        }
    }

    // The MMA_F16 prefill kernel below consumes f16 K and V, and launch_fattn() materialises
    // whichever of them is not already f16 into the scratch past the end of KQV (see
    // ggml_cuda_flash_attn_ext_get_f16_extra_data()). That scratch only exists when the
    // unstreamed kernel selection also asked for f16, and a pair that is direct can still be
    // routed to VEC (a short query batch on Volta: Q->ne[1]*gqa_ratio_eff <= 2) or to the Volta
    // Q8 tensor-core kernel for this very shape - borrowing the MMA kernel then would convert
    // past the end of the KQV allocation. Ask the same question the graph allocator asks.
    bool need_f16_K = false;
    bool need_f16_V = false;
    ggml_cuda_flash_attn_ext_needs_f16(dst, &need_f16_K, &need_f16_V);
    const bool f16_scratch_reserved = need_f16_K && need_f16_V;

    // The MMA prefill span calls launch_fattn() with need_f16_K/V = true, so its inputs are
    // converted into the reserved f16 scratch exactly like the unstreamed MMA_F16 route does
    // (ggml_get_to_fp16_cuda/_nc_cuda cover the turbo types too). What it needs is for that
    // scratch to exist, hence f16_scratch_reserved: a direct pair whose unstreamed kernel
    // would have been VEC never reserved it. Exception: pairs whose TYPED turbo MMA partial
    // kernel is compiled (fattn-mma-turbo -partial instances) consume the raw staged blocks
    // directly and need no f16 scratch at all.
    const bool turbo_mma_partial = ggml_cuda_flash_attn_ext_mma_turbo_partial_available(K->type, V->type);
    const bool use_mma_prefill = kv_stream_mma_prefill_enabled() && !convert_to_f16 &&
        (f16_scratch_reserved || turbo_mma_partial) &&
        Q->ne[1] > 1 && Q->ne[0] == 256 && V->ne[0] == 256 &&
        mask != nullptr && Q->ne[2] % K->ne[2] == 0 && Q->ne[2]/K->ne[2] <= 8;
    const int partial_count = use_mma_prefill ? 1 : kv_stream_parts_per_chunk();
    GGML_ASSERT(partial_count > 0 && partial_count <= KV_STREAM_MAX_PARTS_PER_CHUNK);
    GGML_ASSERT(Q->ne[1] > 0 && nrows % Q->ne[1] == 0);
    const int64_t rows_per_query = nrows/Q->ne[1];
    const int64_t workspace_queries = use_mma_prefill ? Q->ne[1] :
        std::min<int64_t>(Q->ne[1], KV_STREAM_QUERY_WORKSPACE_TOKENS);
    const size_t workspace_rows = size_t(workspace_queries*rows_per_query);
    const size_t workspace_elements = workspace_rows*dst->ne[0];

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> parts(pool);
    ggml_cuda_pool_alloc<float2> meta(pool);
    ggml_cuda_pool_alloc<float> accumulator(pool);
    ggml_cuda_pool_alloc<float2> accumulator_meta(pool);
    const bool needs_partial_reduction = convert_to_f16 || (!streamed_chunks.empty() && nchunks > 1);
    if (needs_partial_reduction) {
        parts.alloc(size_t(partial_count)*workspace_elements);
        meta.alloc(size_t(partial_count)*workspace_rows);
        accumulator.alloc(ggml_nelements(dst));
        accumulator_meta.alloc(nrows);
    }

    auto upload = [&](const chunk_descriptor & desc, cudaStream_t stream) {
        if (!desc.upload) {
            return;
        }
        GGML_ASSERT(K->nb[1] == desc.k_stage_token_stride && K->nb[2] == desc.k_stage_head_stride);
        GGML_ASSERT(V->nb[1] == desc.v_stage_token_stride && V->nb[2] == desc.v_stage_head_stride);
        size_t dirty_row_count = 0;
        if (resident_cache != nullptr && desc.resident_refresh &&
                !resident_cache->all_pages_mutable) {
            for (const int64_t row : resident_cache->dirty_rows) {
                if (row >= desc.token_begin && row < desc.token_begin + desc.token_count) {
                    ++dirty_row_count;
                }
            }
        }
        const size_t dirty_bytes = dirty_row_count*(desc.k_stage_token_stride + desc.v_stage_token_stride);
        if (dirty_row_count > 0 && dirty_bytes < desc.k_bytes + desc.v_bytes) {
            for (const int64_t row : resident_cache->dirty_rows) {
                if (row < desc.token_begin || row >= desc.token_begin + desc.token_count) {
                    continue;
                }
                const size_t page_row = size_t(row - desc.token_begin);
                CUDA_CHECK(cudaMemcpyAsync(
                    desc.k_stage + page_row*desc.k_stage_token_stride,
                    static_cast<const char *>(K->data) + row*K->nb[1],
                    desc.k_stage_token_stride, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    desc.v_stage + page_row*desc.v_stage_token_stride,
                    static_cast<const char *>(V->data) + row*V->nb[1],
                    desc.v_stage_token_stride, cudaMemcpyHostToDevice, stream));
            }
            transfer_ring->host_to_device_copy_commands += 2*dirty_row_count;
            resident_cache->stats.host_to_device_bytes += dirty_bytes;
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            desc.k_stage, static_cast<const char *>(K->data) + desc.token_begin*K->nb[1],
            desc.k_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            desc.v_stage, static_cast<const char *>(V->data) + desc.token_begin*V->nb[1],
            desc.v_bytes, cudaMemcpyHostToDevice, stream));
        transfer_ring->host_to_device_copy_commands += 2;
        if (resident_cache != nullptr) {
            resident_cache->stats.host_to_device_bytes += desc.k_bytes + desc.v_bytes;
        }
    };

    auto schedule_streamed = [&](size_t stream_index) {
        auto & desc = chunks[streamed_chunks[stream_index]];
        const uint32_t slot = desc.slot;
        if (transfer_ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(
                transfer_ring->copy_stream, transfer_ring->consumed[slot], 0));
            ++transfer_ring->stage_slot_reuses;
        }
        upload(desc, transfer_ring->copy_stream);
        CUDA_CHECK(cudaEventRecord(transfer_ring->ready[slot], transfer_ring->copy_stream));
        transfer_ring->slot_used[slot] = 1;
        ++transfer_ring->asynchronous_page_uploads;
    };

    if (!graph_planned && !streamed_chunks.empty()) {
        // SET_ROWS and all other producers for this layer are ordered before
        // this marker on the compute stream. The copy stream may then run
        // independently while attention consumes previously prepared pages.
        CUDA_CHECK(cudaEventRecord(transfer_ring->producer_ready, ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(
            transfer_ring->copy_stream, transfer_ring->producer_ready, 0));
        const size_t initial = std::min<size_t>(
            transfer_ring->active_slots, streamed_chunks.size());
        for (size_t i = 0; i < initial; ++i) {
            schedule_streamed(i);
        }
    }

    size_t stream_index = 0;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        uint32_t streamed_span_pages = 0;
        if (desc.streamed) {
            streamed_span_pages = 1;
            uint32_t ready_slot = desc.slot;
            if (graph_planned) {
                auto & request = transfer_ring->graph_requests[desc.request_index];
                GGML_ASSERT(request.scheduled && !request.consumed);
                desc.slot = request.slot;
                ready_slot = request.ready_slot;
                desc.k_stage = transfer_ring->pool_data + size_t(desc.slot)*desc.k_bytes;
                desc.v_stage = transfer_ring->pool_data +
                    size_t(transfer_ring->active_slots)*desc.k_bytes +
                    size_t(desc.slot)*desc.v_bytes;
                if (request.deadline_sample) {
                    kv_stream_record_deadline<<<1, 1, 0, ctx.stream()>>>(
                        transfer_ring->ready_flags_device + desc.slot,
                        transfer_ring->deadline_counters_device + 0,
                        transfer_ring->deadline_counters_device + 1);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), transfer_ring->ready[ready_slot], 0));
            ++transfer_ring->compute_stream_waits;

            // Coalesce ready pages that occupy consecutive plane slots. The
            // head stride remains the full active-ring plane width, while the
            // tensor's token extent grows across adjacent slots.
            while (!convert_to_f16 &&
                    streamed_span_pages < maximum_streamed_span_pages &&
                    chunk + int(streamed_span_pages) < nchunks) {
                auto & candidate = chunks[chunk + streamed_span_pages];
                uint32_t candidate_ready_slot = candidate.slot;
                if (!candidate.streamed) {
                    break;
                }
                if (graph_planned) {
                    auto & request = transfer_ring->graph_requests[candidate.request_index];
                    if (!request.scheduled || request.consumed) {
                        break;
                    }
                    candidate.slot = request.slot;
                    candidate_ready_slot = request.ready_slot;
                    candidate.k_stage = transfer_ring->pool_data +
                        size_t(candidate.slot)*candidate.k_bytes;
                    candidate.v_stage = transfer_ring->pool_data +
                        size_t(transfer_ring->active_slots)*candidate.k_bytes +
                        size_t(candidate.slot)*candidate.v_bytes;
                }
                if (candidate.slot != desc.slot + streamed_span_pages ||
                        candidate.token_begin != desc.token_begin +
                            int64_t(streamed_span_pages)*block_tokens) {
                    break;
                }
                if (graph_planned) {
                    auto & request = transfer_ring->graph_requests[candidate.request_index];
                    if (request.deadline_sample) {
                        kv_stream_record_deadline<<<1, 1, 0, ctx.stream()>>>(
                            transfer_ring->ready_flags_device + candidate.slot,
                            transfer_ring->deadline_counters_device + 0,
                            transfer_ring->deadline_counters_device + 1);
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
                if (candidate_ready_slot != ready_slot) {
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ctx.stream(), transfer_ring->ready[candidate_ready_slot], 0));
                    ++transfer_ring->compute_stream_waits;
                    ready_slot = candidate_ready_slot;
                }
                ++streamed_span_pages;
            }
            desc.token_count = int64_t(streamed_span_pages)*block_tokens;
            if (resident_cache != nullptr) {
                ++resident_cache->stats.streamed_attention_spans;
                resident_cache->stats.streamed_pages_attended += streamed_span_pages;
            }
        } else {
            if (convert_to_f16) {
                // Generic quantized K/V is converted one page at a time into
                // the bounded workspace. Keep resident pages separate so the
                // fallback never creates a context-sized F16 allocation.
                upload(desc, ctx.stream());
                if (desc.upload && resident_cache->precise_dirty_tracking[resident_layer]) {
                    resident_cache->dirty[kv_stream_resident_index(
                        resident_cache, resident_layer, uint32_t(chunk))] = 0;
                }
                ++resident_cache->stats.resident_attention_spans;
                ++resident_cache->stats.resident_pages_attended;
            } else {
                // Native kernels consume the resident K/V prefix in one span.
                if (chunk > 0) {
                    continue;
                }
                const uint32_t resident_span_pages =
                    std::min<uint32_t>(uint32_t(nchunks), resident_layer_pages);
                GGML_ASSERT(resident_span_pages > 0);
                for (uint32_t page = 0; page < resident_span_pages; ++page) {
                    upload(chunks[page], ctx.stream());
                    if (chunks[page].upload &&
                            resident_cache->precise_dirty_tracking[resident_layer]) {
                        resident_cache->dirty[
                            kv_stream_resident_index(resident_cache, resident_layer, page)] = 0;
                    }
                }
                desc.token_count = int64_t(resident_span_pages)*block_tokens;
                desc.k_head_bytes = desc.k_row_bytes*desc.token_count;
                desc.k_bytes = desc.k_head_bytes*K->ne[2];
                desc.v_head_bytes = desc.v_row_bytes*desc.token_count;
                desc.v_bytes = desc.v_head_bytes*V->ne[2];
                ++resident_cache->stats.resident_attention_spans;
                resident_cache->stats.resident_pages_attended += resident_span_pages;
            }
        }

        ggml_tensor staged_k = *K;
        ggml_tensor staged_v = *V;
        staged_k.data = desc.k_stage;
        staged_k.ne[1] = desc.token_count;
        staged_k.nb[1] = desc.k_stage_token_stride;
        staged_k.nb[2] = desc.k_stage_head_stride;
        staged_k.nb[3] = desc.k_stage_token_stride*desc.token_count;
        staged_v.data = desc.v_stage;
        staged_v.ne[1] = desc.token_count;
        staged_v.nb[1] = desc.v_stage_token_stride;
        staged_v.nb[2] = desc.v_stage_head_stride;
        staged_v.nb[3] = desc.v_stage_token_stride*desc.token_count;

        ggml_tensor converted_k{};
        ggml_tensor converted_v{};
        if (convert_to_f16) {
            GGML_ASSERT(transfer_ring->conversion_data != nullptr);
            const size_t k_elements = size_t(K->ne[0])*desc.token_count*K->ne[2];
            const size_t v_elements = size_t(V->ne[0])*desc.token_count*V->ne[2];
            const size_t k_f16_bytes = k_elements*sizeof(half);
            const size_t v_f16_offset = GGML_PAD(k_f16_bytes, 128);
            const size_t v_f16_bytes = v_elements*sizeof(half);
            GGML_ASSERT(v_f16_offset <= transfer_ring->conversion_bytes &&
                v_f16_bytes <= transfer_ring->conversion_bytes - v_f16_offset);

            auto convert_page = [&](const ggml_tensor & src, half * converted,
                    size_t elements, to_fp16_cuda_t converter) {
                if (converter == nullptr) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        converted, src.data, elements*sizeof(half),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
                    return;
                }
                converter(src.data, converted, elements, ctx.stream());
            };

            auto * converted_k_data = reinterpret_cast<half *>(transfer_ring->conversion_data);
            auto * converted_v_data = reinterpret_cast<half *>(
                transfer_ring->conversion_data + v_f16_offset);
            convert_page(staged_k, converted_k_data, k_elements, converter_k);
            convert_page(staged_v, converted_v_data, v_elements, converter_v);

            converted_k = staged_k;
            converted_k.type = GGML_TYPE_F16;
            converted_k.data = converted_k_data;
            converted_k.nb[0] = sizeof(half);
            converted_k.nb[1] = size_t(K->ne[0])*K->ne[2]*sizeof(half);
            converted_k.nb[2] = size_t(K->ne[0])*sizeof(half);
            converted_k.nb[3] = converted_k.nb[1]*converted_k.ne[1];
            converted_v = staged_v;
            converted_v.type = GGML_TYPE_F16;
            converted_v.data = converted_v_data;
            converted_v.nb[0] = sizeof(half);
            converted_v.nb[1] = size_t(V->ne[0])*V->ne[2]*sizeof(half);
            converted_v.nb[2] = size_t(V->ne[0])*sizeof(half);
            converted_v.nb[3] = converted_v.nb[1]*converted_v.ne[1];
        }
        ggml_tensor staged_mask{};
        ggml_tensor * staged_mask_ptr = nullptr;
        if (mask != nullptr) {
            staged_mask = *mask;
            staged_mask.data = static_cast<char *>(mask->data) + desc.token_begin*mask->nb[0];
            staged_mask.ne[0] = desc.token_count;
            staged_mask_ptr = &staged_mask;
        }

        ggml_tensor staged_dst = *dst;
        staged_dst.src[1] = convert_to_f16 ? &converted_k : &staged_k;
        staged_dst.src[2] = convert_to_f16 ? &converted_v : &staged_v;
        staged_dst.src[3] = staged_mask_ptr;

        // Preserve normal CUDA flash attention when the active cache is fully resident or fits in one streamed page.
        // This avoids a partial reduction and keeps logits identical to a non-streamed cache.
        if (!convert_to_f16 && (streamed_chunks.empty() || nchunks == 1)) {
            ggml_cuda_flash_attn_ext(ctx, &staged_dst);
            if (desc.streamed) {
                if (graph_planned) {
                    kv_stream_graph_release(transfer_ring, desc.request_index, ctx.stream());
                    kv_stream_graph_fill_free_slots(transfer_ring);
                } else {
                    CUDA_CHECK(cudaEventRecord(transfer_ring->consumed[desc.slot], ctx.stream()));
                }
            }
            return;
        }

        GGML_ASSERT(needs_partial_reduction);

        // Keep each staged KV span alive while all bounded query tiles consume it.
        // This bounds vector scratch without issuing another H2D transfer for the span.
        for (int64_t query_begin = 0; query_begin < Q->ne[1]; query_begin += workspace_queries) {
            const int64_t query_count = std::min<int64_t>(
                workspace_queries, Q->ne[1] - query_begin);
            ggml_tensor query_q = *Q;
            query_q.data = static_cast<char *>(Q->data) + query_begin*Q->nb[1];
            query_q.ne[1] = query_count;

            ggml_tensor query_mask{};
            ggml_tensor * query_mask_ptr = nullptr;
            if (staged_mask_ptr != nullptr) {
                query_mask = *staged_mask_ptr;
                query_mask.data = static_cast<char *>(staged_mask_ptr->data) +
                    query_begin*mask->nb[1];
                query_mask.ne[1] = query_count;
                query_mask_ptr = &query_mask;
            }

            ggml_tensor query_dst = staged_dst;
            query_dst.ne[1] = query_count;
            query_dst.src[0] = &query_q;
            query_dst.src[3] = query_mask_ptr;

            if (use_mma_prefill) {
                // Typed turbo MMA partial kernel: decodes the raw staged turbo blocks directly.
                // Only pairs whose -partial instance was compiled reach this; everything else
                // (including all f16-converted routes) falls back to the generic f16 kernel.
                if (!ggml_cuda_flash_attn_ext_mma_turbo_partial_case_typed<256, 256, 8, 8>(
                        ctx, &query_dst, parts.ptr, meta.ptr)) {
                    ggml_cuda_flash_attn_ext_mma_f16_partial_case<256, 256, 8, 8>(
                        ctx, &query_dst, parts.ptr, meta.ptr);
                }
                if (resident_cache != nullptr) {
                    ++resident_cache->stats.mma_prefill_attention_spans;
                }
            } else {
                if (convert_to_f16) {
                    ggml_cuda_flash_attn_ext_vec_partial_case<
                        KV_STREAM_HEAD_DIM, GGML_TYPE_F16, GGML_TYPE_F16>(
                            ctx, &query_dst, parts.ptr, meta.ptr, partial_count);
                } else {
                    GGML_ASSERT(native_partial != nullptr);
                    native_partial(ctx, &query_dst, parts.ptr, meta.ptr, partial_count);
                }
            }

            const int tile_nrows = int(query_count*rows_per_query);
            const size_t row_offset = size_t(query_begin*rows_per_query);
            const dim3 blocks(tile_nrows, 1, 1);
            const dim3 threads(KV_STREAM_HEAD_DIM, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
            ggml_cuda_kernel_launch(
                kv_stream_accumulate_chunk_results<KV_STREAM_HEAD_DIM>, launch_params,
                parts.ptr, meta.ptr, accumulator.ptr + row_offset*dst->ne[0],
                accumulator_meta.ptr + row_offset, tile_nrows, chunk == 0, partial_count);
            CUDA_CHECK(cudaGetLastError());
        }

        if (desc.streamed) {
            for (uint32_t page = 0; page < streamed_span_pages; ++page) {
                auto & member = chunks[chunk + page];
                if (graph_planned) {
                    kv_stream_graph_release(
                        transfer_ring, member.request_index, ctx.stream());
                } else {
                    CUDA_CHECK(cudaEventRecord(
                        transfer_ring->consumed[member.slot], ctx.stream()));
                    const size_t next = stream_index + page + transfer_ring->active_slots;
                    if (next < streamed_chunks.size()) {
                        schedule_streamed(next);
                    }
                }
            }
            stream_index += streamed_span_pages;
            if (graph_planned) {
                kv_stream_graph_fill_free_slots(transfer_ring);
            }
            chunk += int(streamed_span_pages) - 1;
        }
    }

    // Diagnostic: print what the chunk merge accumulated, so the span's contract can be read off
    // instead of guessed. With GGML_CUDA_KV_STREAM_TRACE_SPAN=1, run the same workload once with
    // the span enabled and once disabled; the vec (disabled) run is the reference for what
    // (max, denominator, numerator) should look like.
    if (kv_stream_span_trace_enabled()) {
        static int traces_left = 4;
        if (traces_left > 0) {
            --traces_left;
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            const size_t rows = std::min<size_t>(2, size_t(nrows));
            float2 acc_meta[2] = {};
            float  acc_num[2*4] = {};
            CUDA_CHECK(cudaMemcpy(acc_meta, accumulator_meta.ptr, rows*sizeof(float2),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(acc_num, accumulator.ptr, rows*4*sizeof(float),
                cudaMemcpyDeviceToHost));
            std::fprintf(stderr,
                "kv-stream merge trace: span=%d nchunks=%d parts=%d nrows=%lld "
                "meta0=(%g,%g) meta1=(%g,%g) num0=(%g,%g,%g,%g) num1=(%g,%g,%g,%g)\n",
                int(use_mma_prefill), int(nchunks), partial_count, (long long) nrows,
                acc_meta[0].x, acc_meta[0].y, acc_meta[1].x, acc_meta[1].y,
                acc_num[0], acc_num[1], acc_num[2], acc_num[3],
                acc_num[4], acc_num[5], acc_num[6], acc_num[7]);
        }
    }

    const dim3 blocks(nrows, 1, 1);
    const dim3 threads(KV_STREAM_HEAD_DIM, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(kv_stream_normalize_chunk_results<KV_STREAM_HEAD_DIM>, launch_params,
        accumulator.ptr, accumulator_meta.ptr, static_cast<float *>(dst->data), nrows);
    CUDA_CHECK(cudaGetLastError());
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    // On Volta the mma kernel body itself rejects a tile narrower than 32 columns:
    // fattn-mma-f16.cuh compiles that case to NO_DEVICE_CODE (the
    // "#ifdef VOLTA_MMA_AVAILABLE / ncols1*ncols2 < 32" guard). So the rungs below that would
    // pick ncols1*ncols2 of 8 or 16 must be skipped on Volta -- selecting them launches a
    // kernel that has no device code for this arch, which surfaces as
    //   "CUDA kernel flash_attn_ext_f16 has no device code compatible with CUDA arch 700"
    // followed by "unspecified launch failure". 32/ncols2 * ncols2 is always 32, so the
    // 32-column tile is the narrowest one Volta accepts and is safe to fall through to.
    const bool requires_ncols_ge_32 = volta_mma_available(cc);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse tile is ncols == ncols2 (8 or 16), below Volta's minimum
        if (!requires_ncols_ge_32 && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (!requires_ncols_ge_32 && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    // Qwen3.8-27B on Volta: use the NInfer-derived 32-column sm70 configuration for
    // small/medium cached-prompt appends. The 64-column compact specialization remains faster
    // for wide appends, so keep the measured crossover at 512 query tokens.
    if constexpr (DKQ == 256 && DV == 256 && ncols2 == 2) {
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const bool qwen38_q8 =
            Q->ne[2] == 24 && Q->ne[3] == 1 &&
            K->ne[0] == 256 && K->ne[2] == 4 && K->ne[3] == 1 &&
            V->ne[0] == 256 && V->ne[2] == 4 && V->ne[3] == 1 &&
            K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
        if (cc == GGML_CUDA_CC_VOLTA && qwen38_q8 && Q->ne[1] > 16 && Q->ne[1] <= 512) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    // Prefer the wider 64-column tile, but only when it actually fits the device's opt-in
    // shared-memory limit. On sm_70 that limit is only 96 KB, and some head dims exceed it at
    // ncols=64 (measured: D=320 needs 135168 B, D=576 needs 100864 B), which would otherwise
    // make cudaFuncSetAttribute fail with cudaErrorInvalidValue. Use the 32-column tile, which
    // fits (D=320: 67584 B, D=576: 60928 B), instead of selecting a tile too wide for the GPU.
    // The limit is queried per device, so archs with more shared memory keep the wide tile.
    if (32/ncols2 >= 1 &&
            ggml_cuda_fattn_mma_f16_shared_bytes(DKQ, DV, 64/ncols2, ncols2, cc) >
                ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // Exact long-context Volta GQA8 geometry (D=256, one K/V head): the generic GQA=8
    // choice (ncols2=8) wastes work on heads that do not exist, while ncols2=2 is faster
    // for cached-prompt appends. Opt-in because the crossover is shape dependent.
    if constexpr (DKQ == 256 && DV == 256) {
        if (cc == GGML_CUDA_CC_VOLTA && use_gqa_opt && gqa_ratio == 8 && K->ne[2] == 1 &&
                K->ne[1] >= 65536 && Q->ne[1] >= 128 && Q->ne[1] <= 1024) {
            if (const char * env = getenv("GGML_CUDA_VOLTA_GQA8_NCOLS2")) {
                if (atoi(env) == 2) {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                    return;
                }
            }
        }
    }

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        case 640: {
            // Padded turbo KV cache for GLM-4.7 Flash (K head_dim=576 zero-padded to 640).
            // D=640 shared memory (Q storage = ncols*(DKQ/2+4)*4) exceeds hardware limit at ncols1>=4.
            // Cap at ncols1=2 (ncols=32): Q=32*324*4=41KB + KV≈37KB = ~78KB total.
            GGML_ASSERT(V->ne[0] == 512);
            if (Q->ne[1] <= 1) {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 1, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 2, 16>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// fork 9l-d4b9w: the GGML_CUDA_TURBO_MMA opt-in, split out of
// ggml_cuda_flash_attn_ext_mma_turbo_selectable() so that "the user asked for the tensor-core
// path" and "this (K,V) pair has compiled turbo instances" are two separate questions. The
// uncompiled-pair fallback in ggml_cuda_get_best_fattn_kernel() needs the first one alone.
static bool ggml_cuda_flash_attn_ext_mma_turbo_enabled(void) {
    static const bool turbo_mma_enabled = [] {
        const char * env = std::getenv("GGML_CUDA_TURBO_MMA");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return turbo_mma_enabled;
}

// fork 9l-d4b9w: the authoritative scope of the opt-in's uncompiled-pair fallback -- "with
// GGML_CUDA_TURBO_MMA=1, which KV codecs must land on the f16 tensor-core kernel rather than
// falling through to the VEC/TILE ladder when their turbo MMA instance is missing?"
//
// The turbo types repeat the list inside ggml_cuda_get_best_fattn_kernel()'s is_turbo_kv lambda,
// which sits in slice 9l-d4b8l's frozen neighbourhood and therefore cannot be hoisted (see
// test_9l_d4b8l_guards.py's `blast` row). test_9l_d4b9w_guards.py pins the two lists equal, and
// pins that each of those types is ALSO ggml_is_quantized(), so the explicit cases are documentation
// plus a drift alarm rather than the only thing standing between turbo and the ladder. The dispatch
// below and the exported query read THIS function, so the scope has exactly one edit site.
//
// bf16 belongs here, and is why the list is explicit. It reports ggml_is_quantized() == false
// (ggml.c: blck_size 1, is_quantized false) while every TURBO* type reports true, so the previous
// `is_turbo_kv || ggml_is_quantized` test let bf16 through -- and bf16 has no turbo pair at all, so
// under the opt-in it was silently served by VEC/TILE, the very confound 9l-d4b9w exists to remove.
// A plain f16 cache must still keep the selection it always had, which is why F16 stays out.
static bool ggml_cuda_fattn_kv_needs_mma_f16_fallback(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return ggml_is_quantized(type);
    }
}

bool ggml_backend_cuda_fattn_mma_turbo_fallback_applies(const enum ggml_type type_k,
                                                       const enum ggml_type type_v) {
    return ggml_cuda_fattn_kv_needs_mma_f16_fallback(ggml_type(type_k)) ||
           ggml_cuda_fattn_kv_needs_mma_f16_fallback(ggml_type(type_v));
}

bool ggml_backend_cuda_fattn_mma_turbo_pair_compiled(const enum ggml_type type_k,
                                                     const enum ggml_type type_v) {
    return ggml_cuda_fattn_mma_turbo_pair_compiled(ggml_type(type_k), ggml_type(type_v));
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    // TurboQuant KV cache types. These instances are always compiled (see
    // ggml_cuda_fattn_vec_instances() in ggml/cmake/common.cmake).
    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO4_0)

    FATTN_VEC_CASES_ALL_D(F16,      TURBO2_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO2_0)

    FATTN_VEC_CASES_ALL_D(F16,      TURBO3_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO3_0)

    FATTN_VEC_CASES_ALL_D(F16,      TURBO4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO4_0)

    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO3_0)

    FATTN_VEC_CASES_ALL_D(TURBO8_0, TURBO8_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO8_0)
    FATTN_VEC_CASES_ALL_D(TURBO8_0, TURBO4_0)

    // TurboQuant TCQ (trellis-coded) vec cases.
    FATTN_VEC_CASES_ALL_D(TURBO3_TCQ, TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO2_TCQ, TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO3_TCQ, TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D(F16,    TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(F16,    TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D(Q8_0,   TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(Q8_0,   TURBO2_TCQ)
    // TurboQuant TCQ1 (1.25 bpv).
    FATTN_VEC_CASES_ALL_D(TURBO1_TCQ, TURBO1_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO3_TCQ, TURBO1_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO2_TCQ, TURBO1_TCQ)
    FATTN_VEC_CASES_ALL_D(F16,    TURBO1_TCQ)
    FATTN_VEC_CASES_ALL_D(Q8_0,   TURBO1_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO1_TCQ)

    // Mixed turbo8 / standard-quant / TCQ pairs (VBR mixed-precision caches): each side's
    // K-dot and V-decode already exist for other pairs, these are combination holes only.
    FATTN_VEC_CASES_ALL_D(TURBO8_0, TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(TURBO8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1,     TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(Q4_1,     TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(Q5_1,     TURBO4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1,     TURBO4_0)

    // fork 9l-d4b9y: the t4 -> t3/t2 transitions. A VBR cache whose K side has bottomed out at
    // t4 while the V side keeps degrading passes through exactly these, and they were the only
    // rung of the ladder with no instance in either tree, so the pair could not be reached by any
    // GGML_CUDA_FA_QUANTS list (`all` included). Forward only: the mirrored t3-t4 / t2-t4 pairs
    // are the ones that would mean K is the coarser side.
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO2_TCQ)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
    BEST_FATTN_KERNEL_MMA_TURBO = 350,
    BEST_FATTN_KERNEL_VOLTA_Q8_W4 = 500,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        // TurboQuant KV cache types (fork): dedicated VEC kernel instances
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        // TurboQuant 8-bit / TCQ KV cache types (buun fork): they have VEC kernel
        // instances too (fattn-vec-instance-<K>-<V>.cu). This function is the gate that
        // decides whether ggml_backend_cuda_supports_op() accepts GGML_OP_FLASH_ATTN_EXT
        // at all -- a type missing here does NOT fall back to an f16 kernel on the GPU,
        // it makes the scheduler run the entire attention (and the KV writes feeding it)
        // on the CPU, which is ~2x slower and pins every CPU core. Keep it in sync with
        // the instance files and with is_turbo_kv below.
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1_TCQ:
            return true;
        default:
            return false;
    }
}

// Opt-in gate for the TurboQuant tensor-core FA path (B+ Stage 1c L1, BUUN_MERGE_PLAN.md
// section 1.4a). Enabled via GGML_CUDA_TURBO_MMA=1 (value-based, 0 = off) for A/B benchmarking
// against the regular kernel selection below. Returns true only for shapes that have compiled
// instances per the pair table in fattn-mma-turbo.cuh (13 symmetric pairs across turbo and
// standard quantized KV types, plus turbo8_0-turbo3_tcq and f16-turbo8_0), head dims 128/256,
// and only when the pair's GGML_CUDA_FA_<K>_<V> macro says the instances were compiled
// (GGML_CUDA_FA_QUANTS). Anything else keeps the regular VEC/TILE/MMA_F16 selection, i.e. an
// uncompiled pair silently falls back.
static bool ggml_cuda_flash_attn_ext_mma_turbo_selectable(const ggml_tensor * dst) {
    static const bool turbo_mma_enabled = [] {
        const char * env = std::getenv("GGML_CUDA_TURBO_MMA");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!turbo_mma_enabled) {
        return false;
    }

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (dst->src[0]->ne[0] != 128 && dst->src[0]->ne[0] != 256) {
        return false; // head dim == DKQ for the non-view shapes we instantiate
    }
    return ggml_cuda_fattn_mma_turbo_pair_compiled(K->type, V->type);
}

// Opt-in TurboQuant MMA FA dispatch counter (BUUN_MERGE_PLAN.md section 10.9). Exported via
// ggml_backend_cuda_get_turbo_mma_fa_invocations() so tests and A/B runs can verify this path
// was actually taken: a delta of 0 around a compute means the regular kernels served the op.
static std::atomic<uint64_t> g_turbo_mma_fa_invocations{0};

uint64_t ggml_backend_cuda_get_turbo_mma_fa_invocations(void) {
    return g_turbo_mma_fa_invocations.load(std::memory_order_relaxed);
}

static void ggml_cuda_flash_attn_ext_mma_turbo(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    g_turbo_mma_fa_invocations.fetch_add(1, std::memory_order_relaxed);

    switch (Q->ne[0]) {
        case 128:
            ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256>(ctx, dst);
            break;
        default:
            // ggml_cuda_flash_attn_ext_mma_turbo_selectable() only lets 128/256 through.
            GGML_ABORT("fatal error: no TurboQuant MMA instances for head dim %d", (int) Q->ne[0]);
    }
}

// ---- Dynamic VBR transcode, Stage 1 (read side) ------------------------------------------------------
// fork 9l-d4b8l: this tree's answer to the fork's VBR decode block (fork fattn.cu
// 1403-1513).  Three fork facilities are absent here and are SUBSTITUTED, not ported:
//
//   * RUNTIME decode codebooks (fork d_turbo_centroids_*_fattn, staged by
//     turbo_vanilla_cb_load_fattn / turbo_tcq_load_kv_decode / turbo1_tcq_load_cb_fattn).
//     The port's books are compile-time (turbo-quant.cuh) and they DIFFER from the
//     fork's -- turbo4 0/16 entries equal (max |d| 3.1e-5), turbo3 0/8 (max |d| 3.0e-3),
//     turbo2 4/4.  Porting the fork's kernels verbatim would decode with the WRONG
//     book, so every element below goes through the port's own element accessor
//     (turbo{1,2,3,4,8}_dequant_element) -- the same call this tree's dequantize.cuh
//     makes.  The WHT sign arrays agree (128/128), which is why the inverse-FWHT pair
//     below IS the fork's, verbatim.
//   * K/V-split TCQ books: the port keeps ONE book per TCQ tier and set_rows encodes
//     both sides with it, so `is_v` selects nothing on this path.
//   * the TCQ decode alpha (fork turbo-tcq-alpha.cuh) and the InnerQ inverse channel
//     scale: the port's decode has neither, so both are the identity here.  A->A stays
//     faithful because stage 2 re-encodes through the port's own set_rows.

// In-place 128-point inverse FWHT spread across 128 threads, one element per thread.
// fork 9l-d4b8l: verbatim from fork fattn.cu 798-825 -- the signs agree (see above) and
// the butterfly is pure geometry, so there is nothing to adapt.
// h=1..16 use intra-warp __shfl_xor_sync (no smem, no syncthreads).
// h=32 and h=64 cross warp boundaries, so we fall back to smem with syncs.
// Caller supplies a __shared__ float[128] scratch buffer.
static __device__ __forceinline__ float fwht128_butterfly_inplace(float val, float * smem) {
    const int tid = threadIdx.x;

    // Intra-warp passes: shuffle xor with stride h, no smem, no sync.
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFFULL, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // h=32 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    // h=64 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

// Pair-pack two adjacent threads' fp16 outputs into a single 32-bit half2 store.
// fork 9l-d4b8l: verbatim from fork fattn.cu 827-837.
static __device__ __forceinline__ void fwht128_store_half(
        float val, half * dst_base) {
    const int tid = threadIdx.x;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFFULL, val, 1);
    if ((tid & 1) == 0) {
        const half2 packed = __floats2half2_rn(val, neighbor);
        *((half2 *)(dst_base + tid)) = packed;
    }
}

// f16 -> f32. fork 9l-d4b8l: verbatim from fork fattn.cu 1408-1412.  In the fork the
// scale carries the V decode-alpha correction; here it is always 1.0 (no decode alpha).
static __global__ void k_vbr_f16_to_f32_scaled(const half * __restrict__ src, float * __restrict__ dst,
                                               int64_t n, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] = __half2float(src[i]) * scale; }
}

// F16 source rows (nb1-strided) -> dense scratch_f16. fork 9l-d4b8l: verbatim from fork
// fattn.cu 1414-1428.  The f16 entry tier stores plain original-domain rows (no
// rotation, no mean-sub), so the "dequant" is a gather; the encode tap then runs
// UNsuppressed for f16 sources (see vbr-transcode.cu).
static __global__ void k_vbr_f16_gather(const char * __restrict__ src, half * __restrict__ dst,
                                        const int64_t ne0, const int64_t n_rows, const size_t nb1) {
    const int64_t row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const half * s = (const half *)(src + (size_t) row * nb1);
    half       * d = dst + row * ne0;
    for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
        d[i] = s[i];
    }
}

// Port-side turbo dequant with inverse FWHT: emits one 128-element rotation group in the
// ORIGINAL domain, which is what the transcode's second stage expects (set_rows rotates
// its input itself).  Every turbo tier in this tree is 128-wide in its stored block
// (QK_TURBO3 / QK_TURBO4 / QK_TURBO8 / QK_TURBO{1,2,3}_TCQ are all 128), so one template
// covers all five; only the codebook read differs, and that read is the port's own
// accessor with norm = 1.0, the group norm being applied after the inverse FWHT -- the
// same split the fork uses.
template <ggml_type T> struct vbr_turbo_dequant_traits;

template <> struct vbr_turbo_dequant_traits<GGML_TYPE_TURBO4_0> {
    using block_t = block_turbo4_0;
    static __device__ __forceinline__ float element(const block_t * blk, const int j) {
        return turbo4_dequant_element(blk, j, 1.0f);
    }
};

template <> struct vbr_turbo_dequant_traits<GGML_TYPE_TURBO8_0> {
    using block_t = block_turbo8_0;
    static __device__ __forceinline__ float element(const block_t * blk, const int j) {
        return turbo8_dequant_element(blk, j, 1.0f);
    }
};

template <> struct vbr_turbo_dequant_traits<GGML_TYPE_TURBO3_TCQ> {
    using block_t = block_turbo3_tcq;
    static __device__ __forceinline__ float element(const block_t * blk, const int j) {
        return turbo3_tcq_dequant_element(blk, j, 1.0f);
    }
};

template <> struct vbr_turbo_dequant_traits<GGML_TYPE_TURBO2_TCQ> {
    using block_t = block_turbo2_tcq;
    static __device__ __forceinline__ float element(const block_t * blk, const int j) {
        return turbo2_tcq_dequant_element(blk, j, 1.0f);
    }
};

template <> struct vbr_turbo_dequant_traits<GGML_TYPE_TURBO1_TCQ> {
    using block_t = block_turbo1_tcq;
    static __device__ __forceinline__ float element(const block_t * blk, const int j) {
        return turbo1_tcq_dequant_element(blk, j, 1.0f);
    }
};

template <ggml_type T>
static __global__ void k_vbr_turbo_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    using block_t = typename vbr_turbo_dequant_traits<T>::block_t;

    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    const int n_blocks = (int)(ne0 / 128);   // every turbo tier in this tree is 128-wide

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_t * blk = (const block_t *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        // signs2 -> inverse butterfly -> signs1, then the group norm: the exact mirror of
        // this tree's turbo_rotate_forward (signs1 -> FWHT -> signs2), whose FWHT folds in
        // the same 1/sqrt(128) that the fork's raw butterfly leaves to the caller.
        float val = fwht128_butterfly_inplace(
            vbr_turbo_dequant_traits<T>::element(blk, tid) * TURBO_WHT_SIGNS2[tid], smem);

        val = val * inv_sqrt_128 * TURBO_WHT_SIGNS1[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// Dequant n_cells rows of a turbo KV tensor (type A) to ORIGINAL-domain dense f32
// [n_cells, ne0], pre-scaled so re-encoding to type B then decoding B reproduces A's
// reconstruction.  Lives in fattn.cu for the same reason the fork's does: the decode
// codebooks are TU-static (__constant__ in turbo-quant.cuh) and this TU reaches them
// through fattn-common.cuh.
//
// fork 9l-d4b8l: `type_B` and `is_v` exist in the fork's signature only to pick the target
// tier's decode alpha and the K/V codebook.  This tree has neither, which leaves the
// scaling identity -- the same reason the fidelity check below cannot fail on them.
void vbr_dequant_turbo_to_f32(const char * src, ggml_type src_type, ggml_type type_B,
                              half * scratch_f16, float * dst_f32,
                              int64_t n_cells, int64_t ne0, size_t nb1,
                              bool is_v, int device, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 128 == 0); // turbo FWHT blocks are 128-wide; the dequant loops ne0/128 blocks per cell
    GGML_UNUSED(type_B);  // no decode alpha in this tree, so the target tier adds no scale
    GGML_UNUSED(is_v);    // one TCQ book per tier, shared by K and V (see the banner above)
    GGML_UNUSED(device);  // no per-device codebook staging to do

    // Treat the tensor as [ne0, n_cells]: one block per cell, 128 threads, ne0/128 FWHT
    // blocks per cell. head/stream dims collapse to 1 (nb2/nb3 unused, blockIdx.y/z == 0).
    const dim3 grid((unsigned) n_cells, 1, 1);
    switch (src_type) {
        case GGML_TYPE_TURBO4_0:
            k_vbr_turbo_dequant_f16_inv_fwht<GGML_TYPE_TURBO4_0><<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_TURBO8_0:
            k_vbr_turbo_dequant_f16_inv_fwht<GGML_TYPE_TURBO8_0><<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_TURBO3_TCQ:
            k_vbr_turbo_dequant_f16_inv_fwht<GGML_TYPE_TURBO3_TCQ><<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_TURBO2_TCQ:
            k_vbr_turbo_dequant_f16_inv_fwht<GGML_TYPE_TURBO2_TCQ><<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_TURBO1_TCQ:
            k_vbr_turbo_dequant_f16_inv_fwht<GGML_TYPE_TURBO1_TCQ><<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_F16:
            k_vbr_f16_gather<<<grid, 128, 0, stream>>>(src, scratch_f16, ne0, n_cells, nb1);
            break;
        default:
            GGML_ABORT("vbr_dequant_turbo_to_f32: unsupported src_type %d", (int) src_type);
    }

    // f16 -> f32.  The fork divides V by the TARGET tier's decode alpha here; with no
    // decode alpha in this tree the factor is 1.0 for both sides and both tiers.
    const int64_t n_elem  = n_cells * ne0;
    const int64_t threads = 256;
    const int64_t blocks  = (n_elem + threads - 1) / threads;
    k_vbr_f16_to_f32_scaled<<<(unsigned) blocks, (unsigned) threads, 0, stream>>>(
        scratch_f16, dst_f32, n_elem, 1.0f);
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
        case 640:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // TurboQuant KV cache types: VEC kernels are instantiated for D in {64, 128, 256} only.
    const auto is_turbo_kv = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 ||
               t == GGML_TYPE_TURBO8_0 || t == GGML_TYPE_TURBO3_TCQ ||
               t == GGML_TYPE_TURBO2_TCQ || t == GGML_TYPE_TURBO1_TCQ;
    };
    if ((is_turbo_kv(K->type) || is_turbo_kv(V->type)) && K->ne[0] % 64 != 0) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

#ifdef GGML_USE_HIP
    // HIP/ROCm: the TILE/MMA/WMMA FA paths allocate large f16 temp buffers for
    // quantized KV types (K_f16, V_f16 in launch_fattn). For SMALL batches (decode)
    // the VEC kernel is preferred: it does inline dequant with zero temp buffer
    // overhead, it natively supports the TurboQuant types, and it produces a
    // HIP-graph-safe op stream (no per-call cudaMalloc/cudaFree during capture).
    // For LARGE batches (prefill) the VEC kernel is far slower (sequential query
    // processing), so we deliberately fall through to the TILE/MMA path which is
    // ~3.4x faster; prefill runs eagerly (not captured) so its f16 temp buffer is
    // allocated/freed raw in launch_fattn without violating graph-capture rules.
    // Limitation: head_dim > 256 cannot use VEC (falls through to TILE).
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel && Q->ne[1] <= 8) {
        return BEST_FATTN_KERNEL_VEC;
    }
#endif // GGML_USE_HIP

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        // fork 9l-d4b9x: the TurboQuant MMA opt-in is not Volta-only. The kernel body has no
        // architecture hardcoding (the (DKQ, DV, ncols1, ncols2, type_K, type_V) template takes
        // `cc` as a runtime argument only to size shared memory against smpbo), and the Volta
        // branch below already dispatches it. What kept it off Volta's successors is the call
        // site living inside that branch -- and the Turing branch returns from four places, so
        // the pick has to go here rather than be hoisted above both.
        //
        // The tile ladder in fattn-mma-turbo.cuh stays Volta-shaped: no rung below ncols 32 and
        // no sparse variant, because the kernel body compiles those to NO_DEVICE_CODE on Volta
        // and the ladder applies that restriction unconditionally. On Turing+ that costs the
        // narrow tiles (small query batches stay on VEC) but the wide rungs are correct as-is.
        if (ggml_cuda_flash_attn_ext_mma_turbo_selectable(dst)) {
            return BEST_FATTN_KERNEL_MMA_TURBO;
        }
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 640 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    // Opt-in validated Qwen3.8-27B target-verification route: ordinary llama q8_0 KV is
    // widened only on shared-memory tile load and consumed by Volta tensor cores.
    if (cc == GGML_CUDA_CC_VOLTA && std::getenv("GGML_CUDA_VOLTA_Q8_FATTN_TC") != nullptr &&
            Q->ne[0] == 256 && Q->ne[1] == 4 && Q->ne[2] == 24 && Q->ne[3] == 1 &&
            K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
            K->ne[0] == 256 && V->ne[0] == 256 && K->ne[2] == 4 && V->ne[2] == 4 &&
            K->ne[3] == 1 && V->ne[3] == 1 && mask != nullptr && max_bias == 0.0f &&
            KQV->src[4] == nullptr) {
        float logit_softcap = 0.0f;
        memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
        if (logit_softcap == 0.0f) {
            return BEST_FATTN_KERNEL_VOLTA_Q8_W4;
        }
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        // Opt-in TurboQuant tensor-core path (GGML_CUDA_TURBO_MMA=1). When enabled it takes
        // precedence over VEC/TILE/MMA_F16 for the shapes with compiled instances, so the
        // env var gives a clean A/B against whatever the regular selection below would pick.
        if (ggml_cuda_flash_attn_ext_mma_turbo_selectable(dst)) {
            return BEST_FATTN_KERNEL_MMA_TURBO;
        }
        // fork 9l-d4b9w: the opt-in means "tensor cores, whatever it takes". A pair whose turbo
        // instances were not compiled (GGML_CUDA_FA_MMA off, or the pair missing from
        // GGML_CUDA_FA_QUANTS) must still land on MMA_F16 -- launch_fattn() materialises K and V
        // into the reserved f16 scratch -- instead of falling through to VEC/TILE. Falling
        // through made the A/B meaningless: the turbo kernel was compared against the vector
        // kernel, and a build that omitted the pair looked identical to a build where the turbo
        // path was merely slower. Scoped to the decoded KV codecs so a plain f16 cache keeps the
        // selection it already had; bf16 is one of them now (it has no turbo pair at all and
        // reports is_quantized() == false, so it was the last one still dropped into the ladder).
        // The scope itself lives in ggml_cuda_fattn_kv_needs_mma_f16_fallback(), which the
        // exported ggml_backend_cuda_fattn_mma_turbo_fallback_applies() also reads, so the
        // dispatch and anything asserting on the scope cannot drift apart.
        if (ggml_cuda_flash_attn_ext_mma_turbo_enabled() &&
                (ggml_cuda_fattn_kv_needs_mma_f16_fallback(K->type) ||
                 ggml_cuda_fattn_kv_needs_mma_f16_fallback(V->type))) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // TQ: RDNA4 fast path for TurboQuant cache types - prefer VEC for quantized K/V at small q-cols
    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff_rdna4 = 1;
        const int ncols2_max_rdna4 = (Q->ne[0] == 576 || Q->ne[0] == 640) ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff_rdna4) == 0 && gqa_ratio_eff_rdna4 < ncols2_max_rdna4) {
            gqa_ratio_eff_rdna4 *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff_rdna4 <= 8) {
            return BEST_FATTN_KERNEL_TILE;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// Whether the unstreamed kernel selection needs f16 copies of K and V, i.e. whether the KQV
// allocation reserves the scratch that ggml_cuda_flash_attn_ext_get_f16_extra_data() hands out.
// That scratch lives past the end of the KQV tensor, so every caller that relies on it - the
// conversion inside launch_fattn(), and the streamed path when it borrows the MMA_F16 prefill
// kernel - must ask this, not re-derive it from the K/V types alone.
static void ggml_cuda_flash_attn_ext_needs_f16(
        const ggml_tensor * dst, bool * need_f16_K, bool * need_f16_V) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    *need_f16_K = false;
    *need_f16_V = false;

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            *need_f16_K = true;
            *need_f16_V = true;
            break;
        // TurboQuant MMA decodes turbo KV tiles in-kernel; nothing is converted to f16.
        case BEST_FATTN_KERNEL_MMA_TURBO:
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            *need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            *need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
        case BEST_FATTN_KERNEL_VOLTA_Q8_W4:
            break;
    }
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    if (ggml_cuda_get_best_fattn_kernel(device, dst) == BEST_FATTN_KERNEL_VOLTA_Q8_W4) {
        return ggml_q8v::get_alloc_size(dst);
    }

    bool need_f16_K = false;
    bool need_f16_V = false;
    ggml_cuda_flash_attn_ext_needs_f16(dst, &need_f16_K, &need_f16_V);

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// fork 9l-d4b9w: per-family dispatch counters. The turbo counter above counts ONE family, so a
// delta of 0 around a compute cannot distinguish "9l-d4b9w routed the pair to the f16 tensor-core
// kernel" from "the pair was dropped into the VEC/TILE ladder" -- both are 0, and both decode
// within tolerance, so no numeric assertion separates them either. These counters are the
// observable that makes the routing claim falsifiable, and they are recorded at the single point
// where the selection is turned into a launch, so they name the kernel that actually ran rather
// than the one that was merely considered.
static std::atomic<uint64_t> g_fattn_kernel_invocations[GGML_CUDA_FATTN_KERNEL_COUNT];

static int ggml_cuda_fattn_kernel_family(const best_fattn_kernel k) {
    switch (k) {
        case BEST_FATTN_KERNEL_NONE:        return GGML_CUDA_FATTN_KERNEL_NONE;
        case BEST_FATTN_KERNEL_VEC:         return GGML_CUDA_FATTN_KERNEL_VEC;
        case BEST_FATTN_KERNEL_TILE:        return GGML_CUDA_FATTN_KERNEL_TILE;
        case BEST_FATTN_KERNEL_MMA_F16:     return GGML_CUDA_FATTN_KERNEL_MMA_F16;
        case BEST_FATTN_KERNEL_MMA_TURBO:   return GGML_CUDA_FATTN_KERNEL_MMA_TURBO;
        case BEST_FATTN_KERNEL_VOLTA_Q8_W4: return GGML_CUDA_FATTN_KERNEL_VOLTA_Q8_W4;
    }
    return GGML_CUDA_FATTN_KERNEL_NONE;
}

uint64_t ggml_backend_cuda_get_fattn_kernel_invocations(const int kernel) {
    if (kernel < 0 || kernel >= GGML_CUDA_FATTN_KERNEL_COUNT) {
        return 0;
    }
    return g_fattn_kernel_invocations[kernel].load(std::memory_order_relaxed);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const best_fattn_kernel best = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    g_fattn_kernel_invocations[ggml_cuda_fattn_kernel_family(best)].fetch_add(1, std::memory_order_relaxed);
    switch (best) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_TURBO:
            ggml_cuda_flash_attn_ext_mma_turbo(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VOLTA_Q8_W4:
            ggml_q8v::launch(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
