#include "ggml-turbo-meansub.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// Process-lifetime immutable table plus a mutex-guarded file loader. Deliberately tiny: the
// table is at most 128 x 4096 x 4 B x 2 = 4 MB, and nothing here touches ggml tensors, so this
// file can live in ggml-base and be used from both the CUDA backend and the graph code.

namespace {

struct meansub_table {
    int n_layer = 0;
    int width   = 0;
    std::vector<float> k;  // [n_layer * width] or empty
    std::vector<float> v;  // [n_layer * width] or empty
};

std::mutex    g_mutex;
meansub_table g_table;   // replaced only on a successful load

bool write_all(FILE * f, const void * p, size_t n) {
    return n == 0 || fwrite(p, 1, n, f) == n;
}

bool read_all(FILE * f, void * p, size_t n) {
    return n == 0 || fread(p, 1, n, f) == n;
}

} // namespace

bool ggml_turbo_meansub_load(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }

    char magic[8] = {0};
    uint32_t n_layer = 0, width = 0, flags = 0, reserved = 0;
    bool ok = read_all(f, magic, sizeof(magic)) &&
              read_all(f, &n_layer, sizeof(n_layer)) &&
              read_all(f, &width,   sizeof(width))   &&
              read_all(f, &flags,   sizeof(flags))   &&
              read_all(f, &reserved, sizeof(reserved));

    ok = ok && memcmp(magic, GGML_TURBO_MEANSUB_MAGIC, 7) == 0;
    ok = ok && n_layer > 0 && n_layer <= GGML_TURBO_MEANSUB_MAX_LAYER;
    ok = ok && width   > 0 && width   <= GGML_TURBO_MEANSUB_MAX_WIDTH;
    ok = ok && (flags & (GGML_TURBO_MEANSUB_FLAG_K | GGML_TURBO_MEANSUB_FLAG_V)) != 0;

    meansub_table t;
    t.n_layer = (int) n_layer;
    t.width   = (int) width;

    const size_t n = (size_t) n_layer * (size_t) width;
    if (ok && (flags & GGML_TURBO_MEANSUB_FLAG_K)) {
        t.k.resize(n);
        ok = read_all(f, t.k.data(), n * sizeof(float));
    }
    if (ok && (flags & GGML_TURBO_MEANSUB_FLAG_V)) {
        t.v.resize(n);
        ok = read_all(f, t.v.data(), n * sizeof(float));
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "%s: '%s' is not a usable TurboQuant mean-sub table\n", __func__, path);
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    g_table = std::move(t);
    fprintf(stderr, "%s: loaded '%s' (n_layer=%d width=%d k=%d v=%d)\n", __func__, path,
            g_table.n_layer, g_table.width, !g_table.k.empty(), !g_table.v.empty());
    return true;
}

void ggml_turbo_meansub_load_env(void) {
    const char * env = getenv("TURBO_MEANSUB");
    if (env == nullptr || env[0] == '\0') {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_table.n_layer > 0) {
            return; // already loaded
        }
    }
    ggml_turbo_meansub_load(env);
}

bool ggml_turbo_meansub_ready(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_table.n_layer > 0;
}

int ggml_turbo_meansub_n_layer(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_table.n_layer;
}

int ggml_turbo_meansub_width(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_table.width;
}

const float * ggml_turbo_meansub_k(int layer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_table.k.empty() || layer < 0 || layer >= g_table.n_layer) {
        return nullptr;
    }
    return g_table.k.data() + (size_t) layer * (size_t) g_table.width;
}

const float * ggml_turbo_meansub_v(int layer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_table.v.empty() || layer < 0 || layer >= g_table.n_layer) {
        return nullptr;
    }
    return g_table.v.data() + (size_t) layer * (size_t) g_table.width;
}

bool ggml_turbo_meansub_write(
        const char * path,
        int n_layer, int width, int flags,
        const float * k, const float * v) {
    if (path == nullptr || n_layer <= 0 || width <= 0) {
        return false;
    }

    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }

    const uint32_t nl = (uint32_t) n_layer, w = (uint32_t) width;
    const uint32_t fl = (uint32_t) flags, res = 0;

    bool ok = write_all(f, GGML_TURBO_MEANSUB_MAGIC, 8) &&
              write_all(f, &nl, sizeof(nl)) &&
              write_all(f, &w,  sizeof(w))  &&
              write_all(f, &fl, sizeof(fl)) &&
              write_all(f, &res, sizeof(res));
    if (ok && (flags & GGML_TURBO_MEANSUB_FLAG_K)) {
        ok = write_all(f, k, (size_t) nl * w * sizeof(float));
    }
    if (ok && (flags & GGML_TURBO_MEANSUB_FLAG_V)) {
        ok = write_all(f, v, (size_t) nl * w * sizeof(float));
    }
    fclose(f);
    return ok;
}


#include <string.h>
#include <stdlib.h>
#include <mutex>
#include <atomic>

// Registry entry shape consumed by the generated data table. Compact storage: only live layers
// are stored ([n_live][max_c]) plus their layer indices; the accessor expands to a dense
// [max_l * max_c] slab (dead layers zero) on first use.
typedef struct {
    const char *  arch;
    int           n_layer;
    int           n_embd;
    int           max_l;
    int           max_c;
    int           k_live;
    int           v_live;
    const int *   klay;   // [k_live] layer indices
    const float * kval;   // [k_live * max_c] flattened
    const int *   vlay;   // [v_live]
    const float * vval;   // [v_live * max_c]
} ggml_meansub_entry;

// Before adding a baked affine-tap table, verify that the model has no learned attention sinks.
// K centering shifts only real-token logits, while a learned sink logit is inserted separately;
// V restoration likewise assumes the real-token attention weights sum to one. Supporting sinks
// requires an explicit sink-aware attention contract, not merely another calibrated mean table.
// Also verify the V-mean coordinate contract when head_dim is not a multiple of 128 and there is
// more than one KV head: encode calibration uses padded per-head rows, while graph restoration
// currently expands the table with compact kv_head*head_dim coordinates. The baked models below
// have aligned head dimensions; a padded-head model needs that mismatch fixed before adding it.
#include "ggml-turbo-meansub-data.inc"   // defines g_meansub_table[] + g_meansub_count

static_assert(
        sizeof(g_meansub_table)/sizeof(g_meansub_table[0]) < GGML_TURBO_MEANSUB_MAX_MODELS,
        "GGML_TURBO_MEANSUB_MAX_MODELS must include the unsupported-model slot");

static std::mutex g_dense_mutex;
static std::atomic<float *> g_dense_k[sizeof(g_meansub_table)/sizeof(g_meansub_table[0])] = {};
static std::atomic<float *> g_dense_v[sizeof(g_meansub_table)/sizeof(g_meansub_table[0])] = {};

GGML_API int ggml_turbo_meansub_model_id(const char * arch, int n_layer, int n_embd) {
    if (!arch || !arch[0]) {
        return 0;
    }
    for (int i = 0; i < g_meansub_count; i++) {
        const ggml_meansub_entry * e = &g_meansub_table[i];
        if (n_layer == e->n_layer && n_embd == e->n_embd && strcmp(arch, e->arch) == 0) {
            return i + 1;
        }
    }
    return 0;
}

static float * expand_dense(const ggml_meansub_entry * e, int kvsel) {
    const int     L      = e->max_l;
    const int     C      = e->max_c;
    const int     nlive  = kvsel ? e->v_live : e->k_live;
    const int *   lay    = kvsel ? e->vlay   : e->klay;
    const float * val    = kvsel ? e->vval   : e->kval;

    float * dense = (float *) calloc((size_t) L * C, sizeof(float));
    if (!dense) {
        return NULL;
    }
    for (int i = 0; i < nlive; i++) {
        const int l = lay[i];
        if (l < 0 || l >= L) {
            continue;
        }
        memcpy(dense + (size_t) l * C, val + (size_t) i * C, (size_t) C * sizeof(float));
    }
    return dense;
}

GGML_API const float * ggml_turbo_meansub_table(
        int model_id, int kvsel, int * out_max_l, int * out_max_c, int * out_live) {
    if (model_id <= 0 || model_id > g_meansub_count) {
        return NULL;
    }
    const int index = model_id - 1;
    const ggml_meansub_entry * e = &g_meansub_table[index];
    if (out_max_l) *out_max_l = e->max_l;
    if (out_max_c) *out_max_c = e->max_c;
    if (out_live)  *out_live  = kvsel ? e->v_live : e->k_live;

    std::atomic<float *> & slot = kvsel ? g_dense_v[index] : g_dense_k[index];
    float * dense = slot.load(std::memory_order_acquire);
    if (!dense) {
        std::lock_guard<std::mutex> lock(g_dense_mutex);
        dense = slot.load(std::memory_order_relaxed);
        if (!dense) {
            dense = expand_dense(e, kvsel);
            slot.store(dense, std::memory_order_release);
        }
    }
    return dense;
}
