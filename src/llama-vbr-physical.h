#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// Pure arithmetic used by the cache-level VMM endpoint projector.  Unlike
// llama-vbr-policy.h, this layer is explicitly allocator-aware: its inputs are the
// physical page set and the deferred-unmap queue of one VMM pool.
namespace llama_vbr_physical {

struct interval {
    size_t begin = 0;
    size_t end   = 0;
};

// Match vmm_pool_unmap(): only pages wholly contained in a queued byte range will
// disappear.  The returned intervals are page-aligned, sorted, disjoint, and merged,
// so overlapping queued releases are never subtracted twice.
inline bool normalize_deferred(
        const std::vector<std::pair<size_t, size_t>> & queued,
        size_t granularity,
        std::vector<interval> & out) {
    out.clear();
    if (granularity == 0) {
        return false;
    }

    out.reserve(queued.size());
    for (const auto & q : queued) {
        if (q.second > std::numeric_limits<size_t>::max() - q.first) {
            return false;
        }
        const size_t raw_end = q.first + q.second;
        if (q.first > std::numeric_limits<size_t>::max() - (granularity - 1)) {
            return false;
        }
        const size_t begin = ((q.first + granularity - 1) / granularity) * granularity;
        const size_t end   = (raw_end / granularity) * granularity;
        if (begin < end) {
            out.push_back({ begin, end });
        }
    }

    std::sort(out.begin(), out.end(), [](const interval & a, const interval & b) {
        return a.begin < b.begin || (a.begin == b.begin && a.end < b.end);
    });

    size_t write = 0;
    for (const interval & item : out) {
        if (write != 0 && item.begin <= out[write - 1].end) {
            out[write - 1].end = std::max(out[write - 1].end, item.end);
        } else {
            out[write++] = item;
        }
    }
    out.resize(write);
    return true;
}

// Query resident bytes in [off, off + len), then remove the resident portion of
// deferred releases.  Query must implement exact page-set intersection, as
// vmm_pool_mapped_in_range does.
template<class Query>
inline bool mapped_after_deferred(
        size_t off,
        size_t len,
        size_t granularity,
        const std::vector<interval> & deferred,
        Query && query,
        size_t & out) {
    if (granularity == 0 || off % granularity != 0 || len % granularity != 0 ||
        len > std::numeric_limits<size_t>::max() - off) {
        return false;
    }

    out = query(off, len);
    if (out > len) {
        return false;
    }
    const size_t end = off + len;
    for (const interval & item : deferred) {
        const size_t begin = std::max(off, item.begin);
        const size_t finish = std::min(end, item.end);
        if (begin >= finish) {
            continue;
        }
        const size_t resident = query(begin, finish - begin);
        if (resident > finish - begin || resident > out) {
            return false;
        }
        out -= resident;
    }
    return true;
}

struct projection {
    int64_t  delta   = 0; // release - growth
    uint64_t release = 0;
    uint64_t growth  = 0;
};

inline bool endpoint_bytes(
        uint64_t row_bytes,
        uint32_t watermark,
        uint64_t slot_span,
        uint64_t granularity,
        uint64_t & out) {
    if (granularity == 0 || slot_span == 0 || slot_span % granularity != 0) {
        return false;
    }
    if (watermark != 0 && row_bytes > std::numeric_limits<uint64_t>::max() / watermark) {
        return false;
    }
    const uint64_t raw = row_bytes * watermark;
    if (raw >= slot_span) {
        out = slot_span;
        return true;
    }
    if (raw > std::numeric_limits<uint64_t>::max() - (granularity - 1)) {
        return false;
    }
    const uint64_t padded = ((raw + granularity - 1) / granularity) * granularity;
    out = std::min(slot_span, padded);
    return true;
}

inline bool add_endpoint(
        projection & out,
        uint64_t current_total,
        uint64_t current_inside,
        uint64_t final_bytes) {
    if (current_inside > current_total || current_inside > final_bytes) {
        return false;
    }
    const uint64_t release = current_total - current_inside;
    const uint64_t growth  = final_bytes - current_inside;
    const uint64_t signed_max = (uint64_t) std::numeric_limits<int64_t>::max();
    if (out.release > signed_max || out.growth > signed_max ||
        release > signed_max - out.release || growth > signed_max - out.growth) {
        return false;
    }
    const int64_t step = (int64_t) release - (int64_t) growth;
    if ((step > 0 && out.delta > std::numeric_limits<int64_t>::max() - step) ||
        (step < 0 && out.delta < std::numeric_limits<int64_t>::min() - step)) {
        return false;
    }
    out.delta   += step;
    out.release += release;
    out.growth  += growth;
    return true;
}

} // namespace llama_vbr_physical
