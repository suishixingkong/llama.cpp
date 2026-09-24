#include "llama-vbr-hard-seal.h"

#include <algorithm>
#include <utility>

bool vbr_classify_hard_seal(
        const std::vector<vbr_degrade_step> & order,
        uint8_t seal_tier,
        vbr_hard_seal_classification & out) noexcept {
    vbr_hard_seal_classification result;
    try {
        result.affected.reserve(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            // A custom order may jump directly below the named floor. Such a
            // step crosses the same seal and must not bypass protection merely
            // because the exact T4 endpoint is absent.
            if (order[i].tier >= seal_tier) {
                result.affected.push_back({
                    order[i].il, order[i].is_v != 0, i });
            }
        }
    } catch (...) {
        return false;
    }
    out = std::move(result);
    return true;
}

const vbr_hard_seal_subject * vbr_hard_seal_subject_for_step(
        const vbr_hard_seal_classification & classification,
        size_t order_ordinal) noexcept {
    for (const auto & subject : classification.affected) {
        if (subject.order_ordinal == order_ordinal) {
            return &subject;
        }
    }
    return nullptr;
}

bool vbr_hard_seal_next_order_step(
        size_t & cursor,
        size_t limit,
        const std::vector<size_t> & deferred,
        std::vector<uint8_t> & attempted,
        size_t & order_ordinal,
        bool & from_deferred) noexcept {
    for (const size_t ordinal : deferred) {
        if (ordinal < attempted.size() && !attempted[ordinal]) {
            attempted[ordinal] = 1;
            order_ordinal = ordinal;
            from_deferred = true;
            return true;
        }
    }
    if (cursor >= limit) {
        return false;
    }
    order_ordinal = cursor++;
    from_deferred = false;
    return true;
}

void vbr_hard_seal_defer_step(
        std::vector<size_t> & deferred,
        size_t order_ordinal,
        std::vector<uint8_t> * attempted) {
    if (std::find(deferred.begin(), deferred.end(), order_ordinal) ==
        deferred.end()) {
        deferred.push_back(order_ordinal);
    }
    if (attempted != nullptr && order_ordinal < attempted->size()) {
        (*attempted)[order_ordinal] = 1;
    }
}

void vbr_hard_seal_retire_step(
        std::vector<size_t> & deferred, size_t order_ordinal) noexcept {
    const auto found = std::find(
        deferred.begin(), deferred.end(), order_ordinal);
    if (found != deferred.end()) {
        deferred.erase(found);
    }
}

void vbr_hard_seal_defer_jumped_steps(
        std::vector<size_t> & deferred,
        const std::vector<size_t> & blocked,
        size_t final_cursor) {
    for (const size_t ordinal : blocked) {
        if (ordinal < final_cursor) {
            vbr_hard_seal_defer_step(deferred, ordinal);
        }
    }
}

bool vbr_hard_seal_take_decode_terminal(
        bool decode_failed, bool & blocked) noexcept {
    const bool result = decode_failed && blocked;
    blocked = false;
    return result;
}
