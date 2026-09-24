#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vbr.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

// TurboQuant/VBR KV-cache support — the CUDA implementation of the backend-agnostic
// interface in ggml-vbr.h (struct/semantics docs live there). libllama reaches these
// through ggml_backend_cuda_vbr_iface (resolved via GGML_VBR_BACKEND_IFACE_PROC), never
// by direct link.
GGML_BACKEND_API const struct ggml_vbr_backend_iface * ggml_backend_cuda_vbr_iface(void);
GGML_BACKEND_API const struct ggml_vbr_cross_domain_iface_v1 *
    ggml_backend_cuda_vbr_cross_domain_iface_v1(void);

GGML_BACKEND_API void ggml_backend_cuda_kv_transcode(ggml_backend_t backend,
                                                     const struct ggml_vbr_transcode_params * params);
GGML_BACKEND_API bool ggml_backend_cuda_kv_cross_domain_reconstruct(
        ggml_backend_t backend,
        const struct ggml_vbr_cross_domain_reconstruct_params * params);
GGML_BACKEND_API void ggml_backend_cuda_kv_stash_capture(ggml_backend_t backend, const struct ggml_tensor * src,
                                                         void * stash_f16, int64_t n_rows, bool is_v);
GGML_BACKEND_API void ggml_backend_cuda_sync_device(int device);
GGML_BACKEND_API void ggml_backend_cuda_vbr_fence_arm(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// VMM pool (works on CUDA and ROCm — HIP maps the cuMem* driver API; *_available reports
// the device flag)
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_available(int device);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_granularity(int device);
GGML_BACKEND_API struct ggml_vbr_vmm_pool * ggml_backend_cuda_vmm_pool_init(int device, size_t va_size);
GGML_BACKEND_API void   ggml_backend_cuda_vmm_pool_free(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API void * ggml_backend_cuda_vmm_pool_base(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_pool_mapped(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API uint64_t ggml_backend_cuda_vmm_pool_residency_epoch(struct ggml_vbr_vmm_pool * pool);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_pool_mapped_in_range(
        struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_pool_map(struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_pool_unmap(struct ggml_vbr_vmm_pool * pool, size_t off, size_t len);
GGML_BACKEND_API void   ggml_backend_cuda_vmm_pool_clear(struct ggml_vbr_vmm_pool * pool);

// wrap externally-managed device memory (e.g. a VMM VA range) as a CUDA backend buffer; the buffer
// does NOT take ownership — freeing it never cudaFree's ptr.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_buffer_from_ptr(int device, void * ptr, size_t size);

// Synchronize a CUDA backend and discard captured graphs whose device
// pointers must not survive a compute-arena repartition.
GGML_BACKEND_API bool ggml_backend_cuda_graph_reset(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// Physical CUDA storage shared by phase-specific compute and KV slices.
typedef struct ggml_backend_cuda_phase_arena * ggml_backend_cuda_phase_arena_t;

GGML_BACKEND_API ggml_backend_cuda_phase_arena_t ggml_backend_cuda_phase_arena_new(
    int device, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_phase_arena_free(
    ggml_backend_cuda_phase_arena_t arena);
GGML_BACKEND_API size_t ggml_backend_cuda_phase_arena_size(
    ggml_backend_cuda_phase_arena_t arena);
GGML_BACKEND_API bool ggml_backend_cuda_phase_arena_set_compute(
    ggml_backend_cuda_phase_arena_t arena, size_t offset, size_t size);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_phase_arena_buffer_type(
    ggml_backend_cuda_phase_arena_t arena);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

// Per-context resources for block-granular KV streaming. The buffer type owns
// authoritative pinned-host tensor storage; the runtime owns CUDA staging.
typedef struct ggml_backend_cuda_kv_stream_runtime * ggml_backend_cuda_kv_stream_runtime_t;

struct ggml_backend_cuda_kv_stream_params {
    int device;
    size_t stage_bytes;
    uint32_t stage_slots;
    size_t pool_bytes;
    size_t conversion_bytes;
    uint32_t resident_layer_count;
    uint32_t page_tokens;
    uint32_t decode_span_pages;
};

struct ggml_backend_cuda_kv_stream_type_capabilities {
    bool classified;
    bool storage;
    bool online_write;
    bool decode_f16;
    // The type can be consumed by a native streamed kernel, for at least one partner type.
    // Whether a concrete (K,V) pair can is a property of that pair and is what
    // ggml_backend_cuda_kv_stream_get_attention_mode() reports.
    bool direct_attention;
    bool requires_initialization;
    bool requires_importance_matrix;
    bool auxiliary;
};

enum ggml_backend_cuda_kv_stream_attention_mode {
    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED = 0,
    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT      = 1,
    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16         = 2,
};

GGML_BACKEND_API struct ggml_backend_cuda_kv_stream_type_capabilities
ggml_backend_cuda_kv_stream_get_type_capabilities(enum ggml_type type);
GGML_BACKEND_API enum ggml_backend_cuda_kv_stream_attention_mode
ggml_backend_cuda_kv_stream_get_attention_mode(enum ggml_type type_k, enum ggml_type type_v);

struct ggml_backend_cuda_kv_stream_stats {
    uint64_t resident_hits;
    uint64_t resident_misses;
    uint64_t streamed_pages;
    uint64_t host_to_device_bytes;
    uint64_t resident_attention_spans;
    uint64_t resident_pages_attended;
    uint64_t streamed_attention_spans;
    uint64_t streamed_pages_attended;
    uint64_t mma_prefill_attention_spans;
    uint64_t asynchronous_page_uploads;
    uint64_t host_to_device_copy_commands;
    uint64_t compute_stream_waits;
    uint64_t stage_slot_reuses;
    uint64_t cross_layer_prefetches;
    uint64_t deadline_samples;
    uint64_t deadline_misses;
    uint32_t ring_peak_occupancy;
    uint64_t staged_set_rows;
    uint64_t staged_set_rows_bytes;
};

GGML_BACKEND_API ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
    struct ggml_backend_cuda_kv_stream_params params);
GGML_BACKEND_API ggml_backend_cuda_kv_stream_runtime_t
ggml_backend_cuda_kv_stream_runtime_new_in_phase_arena(
    ggml_backend_cuda_phase_arena_t arena,
    size_t maximum_pool_bytes,
    struct ggml_backend_cuda_kv_stream_params params);
GGML_BACKEND_API void ggml_backend_cuda_kv_stream_runtime_free(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API size_t ggml_backend_cuda_kv_stream_stage_bytes(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API uint32_t ggml_backend_cuda_kv_stream_stage_slots(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API uint32_t ggml_backend_cuda_kv_stream_resident_pages_per_layer(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API size_t ggml_backend_cuda_kv_stream_pool_bytes(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_resize_pool(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    size_t pool_bytes,
    uint32_t active_pages_per_layer,
    uint32_t stage_slots);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_reconfigure(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t active_pages_per_layer,
    uint32_t stage_slots);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_repartition(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t stage_slots);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_set_decode_layout(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t active_pages_per_layer);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_mark_dirty_rows(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    const int64_t * rows,
    size_t count);
GGML_BACKEND_API struct ggml_backend_cuda_kv_stream_stats ggml_backend_cuda_kv_stream_get_stats(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_stage_upload(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t slot,
    size_t offset,
    const void * source,
    size_t size);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_stage_download(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t slot,
    size_t offset,
    void * destination,
    size_t size);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// Number of FLASH_ATTN_EXT ops dispatched to the opt-in TurboQuant MMA kernel
// (GGML_CUDA_TURBO_MMA=1) since library start. Read a delta around a compute to verify the
// path actually taken: a delta of 0 means the regular VEC/TILE/MMA_F16 kernels served the op
// (e.g. the shape had no compiled instance, or the env var was off) -- a silent fallback
// would otherwise make a "turbo MMA" benchmark measure the regular kernels.
GGML_BACKEND_API uint64_t ggml_backend_cuda_get_turbo_mma_fa_invocations(void);

// #88: boundary-time reserve of one backend context's fattn f16 dequant scratch
// (ggml-vbr.h vtable slot)
GGML_BACKEND_API bool ggml_backend_cuda_kv_dequant_scratch_reserve(
        ggml_backend_t backend, size_t k_bytes, size_t v_bytes);
GGML_BACKEND_API void ggml_backend_cuda_kv_dequant_scratch_memory(
        ggml_backend_t backend, size_t k_bytes, size_t v_bytes,
        size_t * physical_now, size_t * physical_if_reserved);
GGML_BACKEND_API bool ggml_backend_cuda_kv_transcode_workspace_memory(
        ggml_backend_t backend_or_null, int device,
        int64_t n_cells, int64_t ne0, int64_t stash_rows,
        size_t * physical_now, size_t * physical_if_reserved);
GGML_BACKEND_API bool ggml_backend_cuda_kv_transcode_workspace_reserve(
        ggml_backend_t backend, int64_t n_cells, int64_t ne0, int64_t stash_rows);
GGML_BACKEND_API bool ggml_backend_cuda_kv_transcode_workspace_memory_v2(
        ggml_backend_t backend_or_null, int device,
        const struct ggml_vbr_transcode_workspace_params_v2 * params,
        size_t * physical_now, size_t * physical_if_reserved);
GGML_BACKEND_API bool ggml_backend_cuda_kv_transcode_workspace_reserve_v2(
        ggml_backend_t backend,
        const struct ggml_vbr_transcode_workspace_params_v2 * params);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
