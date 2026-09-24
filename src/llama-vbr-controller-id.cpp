#include "llama-vbr-controller-id.h"

#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <random>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr uint64_t VBR_CONTROLLER_INSTANCE_DOMAIN = UINT64_C(0x564252494e535431);
constexpr size_t   VBR_CONTROLLER_REGISTRY_CAPACITY = 4096;

struct controller_registry_entry {
    vbr_controller_instance_id instance = {};
    const void *               owner    = nullptr;
};

std::atomic<uint64_t> g_lineage_counter { 1 };
std::atomic<bool>     g_lineage_exhausted { false };
std::mutex            g_lineage_origin_mutex;
uint64_t              g_lineage_origin = 0;
bool                  g_lineage_origin_initialized = false;
vbr_lineage_origin_provider g_lineage_origin_provider = nullptr;

std::atomic<uint64_t> g_instance_counter { 1 };
std::atomic<bool>     g_instance_exhausted { false };

std::mutex g_controller_registry_mutex;
std::array<controller_registry_entry, VBR_CONTROLLER_REGISTRY_CAPACITY>
    g_controller_registry = {};
size_t g_controller_registry_capacity = VBR_CONTROLLER_REGISTRY_CAPACITY;

uint64_t mix64(uint64_t value) noexcept {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

bool production_lineage_origin(uint64_t & origin) noexcept {
    try {
        std::random_device random;
        std::array<uint64_t, 4> words = {};
        bool any_random_bit = false;
        for (auto & word : words) {
            word = (uint64_t(random()) << 32) ^ uint64_t(random());
            any_random_bit = any_random_bit || word != 0;
        }
        if (!any_random_bit) {
            return false;
        }

        const uint64_t clock = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
#if defined(_WIN32)
        const uint64_t pid = static_cast<uint64_t>(_getpid());
#else
        const uint64_t pid = static_cast<uint64_t>(getpid());
#endif
        static const uint8_t aslr_marker = 0;
        const uint64_t address = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(&aslr_marker));

        uint64_t mixed = mix64(words[0] ^ UINT64_C(0x6c696e6561676531));
        mixed = mix64(mixed ^ words[1]);
        mixed = mix64(mixed ^ words[2]);
        mixed = mix64(mixed ^ words[3]);
        mixed = mix64(mixed ^ clock);
        mixed = mix64(mixed ^ pid);
        mixed = mix64(mixed ^ address);
        if (mixed == 0) {
            return false;
        }
        origin = mixed;
        return true;
    } catch (...) {
        return false;
    }
}

template<typename Id>
Id allocate_counter_id(std::atomic<uint64_t> & counter,
                       std::atomic<bool> & exhausted,
                       uint64_t opaque_high_word) noexcept {
    if (exhausted.load(std::memory_order_acquire)) {
        return {};
    }
    uint64_t expected = counter.load(std::memory_order_relaxed);
    for (;;) {
        if (expected == 0) {
            exhausted.store(true, std::memory_order_release);
            return {};
        }
        const uint64_t next = expected == std::numeric_limits<uint64_t>::max()
                                  ? 0
                                  : expected + 1;
        if (counter.compare_exchange_weak(
                expected, next, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            if (next == 0) {
                exhausted.store(true, std::memory_order_release);
            }
            return { opaque_high_word, expected };
        }
    }
}

bool controller_registry_empty_locked() noexcept {
    for (size_t i = 0; i < g_controller_registry_capacity; ++i) {
        if (g_controller_registry[i].owner != nullptr) {
            return false;
        }
    }
    return true;
}

size_t find_owned_slot_locked(
        vbr_controller_instance_id instance,
        size_t * first_empty = nullptr) noexcept {
    const size_t none = g_controller_registry_capacity;
    if (first_empty != nullptr) {
        *first_empty = none;
    }
    for (size_t i = 0; i < g_controller_registry_capacity; ++i) {
        const auto & entry = g_controller_registry[i];
        if (entry.owner == nullptr) {
            if (first_empty != nullptr && *first_empty == none) {
                *first_empty = i;
            }
        } else if (entry.instance == instance) {
            return i;
        }
    }
    return none;
}

} // namespace

vbr_lineage_uuid vbr_lineage_uuid_allocate() noexcept {
    uint64_t origin = 0;
    {
        std::lock_guard<std::mutex> lock(g_lineage_origin_mutex);
        if (!g_lineage_origin_initialized) {
            const auto provider = g_lineage_origin_provider != nullptr
                                      ? g_lineage_origin_provider
                                      : production_lineage_origin;
            uint64_t candidate = 0;
            if (!provider(candidate) || candidate == 0) {
                g_lineage_exhausted.store(true, std::memory_order_release);
                g_lineage_origin_initialized = true;
                return {};
            }
            g_lineage_origin = candidate;
            g_lineage_origin_initialized = true;
        }
        origin = g_lineage_origin;
    }
    return allocate_counter_id<vbr_lineage_uuid>(
        g_lineage_counter, g_lineage_exhausted, origin);
}

vbr_controller_instance_id vbr_controller_instance_id_allocate() noexcept {
    return allocate_counter_id<vbr_controller_instance_id>(
        g_instance_counter, g_instance_exhausted,
        VBR_CONTROLLER_INSTANCE_DOMAIN);
}

bool vbr_controller_instance_check_and_claim(
        vbr_controller_instance_id instance,
        const void * owner) noexcept {
    if (!vbr_controller_instance_id_is_set(instance) || owner == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_controller_registry_mutex);
    size_t first_empty = g_controller_registry_capacity;
    if (find_owned_slot_locked(instance, &first_empty) !=
            g_controller_registry_capacity ||
        first_empty == g_controller_registry_capacity) {
        return false;
    }
    g_controller_registry[first_empty] = { instance, owner };
    return true;
}

bool vbr_controller_instance_release(
        vbr_controller_instance_id instance,
        const void * owner) noexcept {
    if (!vbr_controller_instance_id_is_set(instance) || owner == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_controller_registry_mutex);
    const size_t slot = find_owned_slot_locked(instance);
    if (slot == g_controller_registry_capacity ||
        g_controller_registry[slot].owner != owner) {
        return false;
    }
    g_controller_registry[slot] = {};
    return true;
}

bool vbr_controller_instance_owned_by(
        vbr_controller_instance_id instance,
        const void * owner) noexcept {
    if (!vbr_controller_instance_id_is_set(instance) || owner == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_controller_registry_mutex);
    const size_t slot = find_owned_slot_locked(instance);
    return slot != g_controller_registry_capacity &&
           g_controller_registry[slot].owner == owner;
}

bool vbr_lineage_origin_provider_set_for_tests(
        vbr_lineage_origin_provider provider) noexcept {
    if (provider == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> registry_lock(g_controller_registry_mutex);
    if (!controller_registry_empty_locked()) {
        return false;
    }
    std::lock_guard<std::mutex> origin_lock(g_lineage_origin_mutex);
    g_lineage_origin_provider = provider;
    g_lineage_origin = 0;
    g_lineage_origin_initialized = false;
    g_lineage_counter.store(1, std::memory_order_release);
    g_lineage_exhausted.store(false, std::memory_order_release);
    return true;
}

bool vbr_controller_instance_registry_capacity_set_for_tests(
        size_t capacity) noexcept {
    if (capacity == 0 || capacity > VBR_CONTROLLER_REGISTRY_CAPACITY) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_controller_registry_mutex);
    if (!controller_registry_empty_locked()) {
        return false;
    }
    g_controller_registry_capacity = capacity;
    return true;
}

size_t vbr_controller_instance_registry_capacity() noexcept {
    std::lock_guard<std::mutex> lock(g_controller_registry_mutex);
    return g_controller_registry_capacity;
}
