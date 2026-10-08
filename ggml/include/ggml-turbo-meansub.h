#pragma once

// TurboQuant affine-tap ("mean-sub") tables: per-(layer, channel) KV means that the CUDA encode
// path subtracts before the WHT rotation, and that the graph restores for V after attention.
// See MERGE_TURBO234_UPSTREAM.md sections 3-5 and MERGE_BUUN_PLAN.md section 10.20.
//
// Why:
//   * K: logits use <q, k - mu_k> = <q, k> - <q, mu_k>, so centring K shifts every real-token
//     logit by the same amount and leaves the softmax output unchanged. Its only effect is to
//     shrink K's dynamic range, which lowers the 4-bit quantisation error. This requires that
//     the model has NO learned attention sink (a sink logit is inserted separately, so a
//     uniform shift of the real-token logits would change the sink's relative weight).
//   * V: y = sum_i w_i (v_i - mu_v) = sum_i w_i v_i - mu_v * sum_i w_i, so the graph must add
//     mu_v back after the attention. This is exact when the real-token attention weights sum
//     to one.
//
// Storage: a small self-describing file, NOT a compiled-in blob, so that calibrating a new
// model does not require rebuilding:
//
//     TQTAPMS1                              magic, 8 bytes
//     u32 n_layer                           <= GGML_TURBO_MEANSUB_MAX_LAYER
//     u32 width                             == n_head_kv * head_dim, <= ..._MAX_WIDTH
//     u32 flags                             bit0 = K present, bit1 = V present
//     u32 reserved
//     f32 k[n_layer][width]                 layer-major, head-major, channel-minor (if bit0)
//     f32 v[n_layer][width]                 (if bit1)
//
// The file is produced by the in-kernel capture tool (set TURBO_MEANSUB_CAPTURE, see
// ggml/src/ggml-cuda/set-rows.cu) and consumed by the encode path / graph restore.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TURBO_MEANSUB_MAGIC      "TQTAPMS1"
#define GGML_TURBO_MEANSUB_MAX_LAYER  128
#define GGML_TURBO_MEANSUB_MAX_WIDTH  4096

// Load a table file. On any error the previously loaded table is left untouched and false is
// returned. Thread-safe; the loaded table is immutable for the process lifetime.
GGML_API bool ggml_turbo_meansub_load(const char * path);

// Load from $TURBO_MEANSUB when set and no table is loaded yet. Idempotent.
GGML_API void ggml_turbo_meansub_load_env(void);

GGML_API bool ggml_turbo_meansub_ready(void);
GGML_API int  ggml_turbo_meansub_n_layer(void);
GGML_API int  ggml_turbo_meansub_width(void);

// Row for one layer (length == ggml_turbo_meansub_width()), or NULL when absent / out of range.
// The row covers n_head_kv * head_dim channels, head-major, so channel c of head h is row[h*head_dim + c].
GGML_API const float * ggml_turbo_meansub_k(int layer);
GGML_API const float * ggml_turbo_meansub_v(int layer);

// Write a table in the format above. Used by the capture tool. k or v may be NULL when the
// corresponding flag bit is clear.
GGML_API bool ggml_turbo_meansub_write(
        const char * path,
        int n_layer, int width, int flags,
        const float * k, const float * v);

#define GGML_TURBO_MEANSUB_FLAG_K 1u
#define GGML_TURBO_MEANSUB_FLAG_V 2u

// --- baked per-architecture calibration registry (buun) ---
#define GGML_TURBO_MEANSUB_MAX_MODELS 16
#define GGML_TURBO_MEANSUB_MAX_L      128
#define GGML_TURBO_MEANSUB_MAX_C      2048

// Resolve a stable registry ID for one model identity. arch = general.architecture string.
// The match requires arch + n_layer + n_embd to agree. Returns 0 when no table matches.
GGML_API int ggml_turbo_meansub_model_id(const char * arch, int n_layer, int n_embd);

// Dense baked mean slab [max_l * max_c] for model_id; kvsel: 0 = K, 1 = V.
// Returns NULL when model_id is 0 or invalid. Returned storage is immutable and process-lifetime;
// out_* may be NULL.
GGML_API const float * ggml_turbo_meansub_table(
        int model_id, int kvsel, int * out_max_l, int * out_max_c, int * out_live);

#ifdef __cplusplus
}
#endif
