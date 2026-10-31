#include "kvmem/host_kv_storage.hpp"
#include "kvmem/memory_contract.hpp"
#include "kvmem/spill_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvmem {
namespace detail {
// A size-class cache in front of the heap. Evicting a block and staging it back
// in is "free N bytes, then allocate N bytes": RawKvStore drops its GPU mirror
// before it can tell whether the old buffer would do, so the same size comes
// back asking for a fresh allocation every time. Reusing the pages keeps them
// mapped, which is where most of the win is -- a fresh 32 KiB block faults in
// eight pages that the following memcpy then fills. KVMEM_HOST_ARENA_MB=0 turns
// this off so the two can be compared on one binary.
struct HostKvArena {
    explicit HostKvArena(size_t budget_bytes) : budget(budget_bytes) {}
    uint8_t * acquire(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = reusable.find(bytes);
        if (it != reusable.end() && !it->second.empty()) {
            uint8_t * block = it->second.back();
            it->second.pop_back();
            cached -= bytes;
            ++hits;
            return block;
        }
        ++misses;
        return nullptr;
    }
    // True when the block was taken; false means the caller keeps it and must
    // hand it back to the heap itself.
    bool release(uint8_t * block, size_t bytes) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (cached + bytes > budget) {
            ++dropped;
            return false;
        }
        reusable[bytes].push_back(block);
        cached += bytes;
        return true;
    }
    std::mutex mutex;
    std::unordered_map<size_t, std::vector<uint8_t *>> reusable;
    size_t budget = 0;
    size_t cached = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t dropped = 0;
};

struct HostKvAccount {
    static constexpr uint64_t write_batch_bytes = 256 * 1024;
    HostKvAccount(uint64_t capacity, std::shared_ptr<SpillFile> spill);
    std::mutex mutex;
    const uint64_t capacity;
    uint64_t charged = 0;
    const std::shared_ptr<SpillFile> spill;
    // Blocks parked for reuse are released from the budget above, so this cache
    // is extra process memory on top of `capacity`: bounded, and small next to a
    // real host quota.
    std::shared_ptr<HostKvArena> arena;
    // Catalog selection never waits on a busy record. Physical writes hold only
    // their record lock, so unrelated resident lanes and reservations can run.
    // RAM-only operations never acquire policy/record locks or populate this list.
    std::mutex policy;
    std::condition_variable relocation_finished;
    size_t relocating = 0;
    std::list<std::weak_ptr<HostKvAllocation>> residents;
    unsigned registrations = 0;
    void register_resident(const std::shared_ptr<HostKvAllocation> & record) {
        if (++registrations == 64) {
            residents.remove_if([](const auto & item) { return item.expired(); });
            registrations = 0;
        }
        residents.push_back(record);
    }
    void release(uint64_t bytes) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        charged -= bytes;
    }
    bool try_charge(uint64_t bytes, uint64_t headroom = 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (bytes > capacity - charged || headroom > capacity - charged - bytes) return false;
        charged += bytes;
        return true;
    }
    uint64_t usage() {
        std::lock_guard<std::mutex> lock(mutex); return charged;
    }
    bool evict_batch(std::unique_lock<std::mutex> & policy_lock, const HostKvAllocation * exclude);
    void reserve_locked(std::unique_lock<std::mutex> & policy_lock, uint64_t bytes,
                        const HostKvAllocation * exclude = nullptr) {
        if (bytes > capacity)
            throw MemoryError(MemoryErrorCode::BudgetExceeded, "native KV allocation exceeds RAM capacity");
        // Prefer room for one bounded packing buffer. This is not a hard
        // reduction of H: mutable/pinned records and large reservations may
        // use the full budget and fall back to single-record writes.
        const auto headroom = capacity >= 2 * write_batch_bytes && bytes <= capacity - write_batch_bytes
            ? write_batch_bytes : 0;
        while (!try_charge(bytes, headroom)) {
            bool evicted;
            try { evicted = evict_batch(policy_lock, exclude); }
            catch (const MemoryError & error) {
                if (error.code() == MemoryErrorCode::BudgetExceeded && try_charge(bytes)) return;
                throw;
            }
            if (!evicted) {
                if (try_charge(bytes)) return;
                // Another CPU worker is releasing H. Wait without owning the
                // catalog lock; its completed I/O will wake this reservation.
                if (relocating) { relocation_finished.wait(policy_lock); continue; }
                throw MemoryError(MemoryErrorCode::BudgetExceeded, "native KV RAM is held by mutable records/leases");
            }
        }
    }
};
struct HostKvBlockDeleter {
    std::shared_ptr<HostKvAccount> account;
    size_t bytes = 0;
    void operator()(uint8_t * block) const noexcept {
        if (block == nullptr) return;
        if (account && account->arena && account->arena->release(block, bytes)) return;
        delete[] block;
    }
};

// Single place that turns a byte count into a host block, so the arena cannot be
// bypassed by one of the two allocation sites. `zero` keeps the old contract of
// `new uint8_t[n]()`: callers that only write part of the block read the rest.
inline std::unique_ptr<uint8_t[], HostKvBlockDeleter> host_kv_alloc_block(
        const std::shared_ptr<HostKvAccount> & account, size_t bytes, bool zero) {
    uint8_t * block = account && account->arena ? account->arena->acquire(bytes) : nullptr;
    if (block == nullptr) block = new uint8_t[bytes];
    if (zero) std::memset(block, 0, bytes);
    return std::unique_ptr<uint8_t[], HostKvBlockDeleter>(block, HostKvBlockDeleter{account, bytes});
}

struct HostKvAllocation {
    std::shared_ptr<HostKvAccount> account;
    std::unique_ptr<uint8_t[], HostKvBlockDeleter> bytes;
    size_t size = 0;
    std::mutex mutex;
    std::atomic<uint32_t> pins{0};
    std::atomic<uint32_t> readers{0};
    std::atomic<bool> sealed{false};
    std::atomic<size_t> resident_size{0};
    bool relocating = false; // protected by account policy
    SpillExtent disk;
    ~HostKvAllocation() {
        const bool resident = bool(bytes);
        bytes.reset();
        if (account && resident) account->release(size);
    }
};
HostKvAccount::HostKvAccount(uint64_t capacity, std::shared_ptr<SpillFile> spill)
    : capacity(capacity), spill(std::move(spill)) {
    // An eighth of the host quota, capped so a huge or unset quota cannot park
    // gigabytes of idle blocks. 0 disables the cache, which is how the two paths
    // are compared on one binary.
    //
    // 128 MiB is where a pp8192 run over 0.8B stops dropping blocks: the live set
    // of 32 KiB size classes settles at ~96 MiB and `dropped` reaches zero, so
    // anything above that buys nothing here and only delays the memory going back
    // to the OS. Bigger models with more layers will want the override.
    uint64_t budget = capacity / 8;
    const uint64_t cap = 128ull * 1024 * 1024;
    if (budget > cap) budget = cap;
    if (const char * e = std::getenv("KVMEM_HOST_ARENA_MB")) {
        const long mb = std::strtol(e, nullptr, 10);
        if (mb >= 0) budget = (uint64_t) mb * 1024 * 1024;
    }
    if (budget > 0) arena = std::make_shared<HostKvArena>((size_t) budget);
}

bool HostKvAccount::evict_batch(std::unique_lock<std::mutex> & policy_lock, const HostKvAllocation * exclude) {
    if (!spill) return false;
    struct Candidate {
        std::shared_ptr<HostKvAllocation> record;
        std::unique_lock<std::mutex> lock;
        std::list<std::weak_ptr<HostKvAllocation>>::iterator position;
    };
    const uint64_t scratch_limit = std::min(write_batch_bytes, capacity - usage());
    std::vector<Candidate> batch;
    std::vector<SpillExtent> destinations;
    size_t total = 0;
    const size_t attempts = residents.size();
    std::exception_ptr capacity_error;
    for (size_t i = 0; i < attempts; ++i) {
        auto position = residents.begin();
        auto record = position->lock();
        if (!record) { residents.erase(position); continue; }
        if (record.get() == exclude || record->relocating) {
            residents.splice(residents.end(), residents, position); continue;
        }
        std::unique_lock<std::mutex> lock(record->mutex, std::try_to_lock);
        if (!lock) { residents.splice(residents.end(), residents, position); continue; }
        if (!record->bytes) { residents.erase(position); continue; }
        if (!record->sealed || record->pins.load(std::memory_order_acquire)) {
            residents.splice(residents.end(), residents, position); continue;
        }
        if (!batch.empty() && record->size > scratch_limit - total) break;
        // Destination is private/unpublished until the whole write succeeds.
        // Any quota/allocation/I/O exception leaves the sole valid RAM copy intact.
        SpillExtent destination;
        try { destination = spill->reserve(record->size); }
        catch (const MemoryError & error) {
            if (error.code() != MemoryErrorCode::BudgetExceeded) throw;
            capacity_error = std::current_exception();
            residents.splice(residents.end(), residents, position);
            continue; // A smaller record may still fit the remaining extent.
        }
        total += record->size;
        destinations.push_back(std::move(destination));
        batch.push_back({std::move(record), std::move(lock), position});
        residents.splice(residents.end(), residents, position);
        // A large record or exhausted scratch space keeps the existing direct
        // write path. Also bound the number of simultaneously held record locks.
        if (total >= scratch_limit || batch.size() == 64) break;
    }
    if (batch.empty()) {
        if (capacity_error && !relocating) std::rethrow_exception(capacity_error);
        return false;
    }
    const bool packed = batch.size() > 1;
    if (packed && !try_charge(total))
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "KV batch staging exceeds RAM capacity");
    for (auto & item : batch) item.record->relocating = true;
    relocating += batch.size();
    policy_lock.unlock();
    try {
        if (packed) {
            // The source remains authoritative and charged until every write
            // completes. Packing memory is charged to the same H account.
            auto scratch = std::unique_ptr<uint8_t[]>(new uint8_t[total]);
            size_t offset = 0;
            for (const auto & item : batch) {
                std::memcpy(scratch.get() + offset, item.record->bytes.get(), item.record->size);
                offset += item.record->size;
            }
            spill->write_batch(destinations, scratch.get(), total);
        } else {
            destinations.front().write(0, batch.front().record->bytes.get(), total);
        }
    } catch (...) {
        if (packed) release(total);
        policy_lock.lock();
        for (auto & item : batch) item.record->relocating = false;
        relocating -= batch.size();
        relocation_finished.notify_all();
        throw;
    }
    if (packed) release(total);
    policy_lock.lock();
    for (size_t i = 0; i < batch.size(); ++i) {
        auto & item = batch[i];
        item.record->relocating = false;
        item.record->disk = std::move(destinations[i]);
        item.record->bytes.reset();
        item.record->resident_size.store(0, std::memory_order_release);
        release(item.record->size);
        residents.erase(item.position);
    }
    relocating -= batch.size();
    relocation_finished.notify_all();
    return true;
}
} // namespace detail
namespace {
void check_range(size_t size, size_t offset, const void * pointer, size_t bytes) {
    if (offset > size || bytes > size - offset || (bytes && !pointer))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "native KV byte range is invalid");
}
void unpin(std::shared_ptr<detail::HostKvAllocation> & allocation) noexcept {
    if (allocation) allocation->pins.fetch_sub(1, std::memory_order_release);
    allocation.reset();
}
void release_source(std::shared_ptr<detail::HostKvAllocation> & allocation) noexcept {
    if (allocation) allocation->readers.fetch_sub(1, std::memory_order_release);
    allocation.reset();
}
} // namespace

HostKvLease::HostKvLease(std::shared_ptr<detail::HostKvAllocation> allocation)
    : allocation_(std::move(allocation)) {
    if (allocation_) allocation_->pins.fetch_add(1, std::memory_order_relaxed);
}
HostKvLease::HostKvLease(const HostKvLease & other) : HostKvLease(other.allocation_) {}
HostKvLease & HostKvLease::operator=(const HostKvLease & other) {
    if (this != &other) { HostKvLease copy(other); *this = std::move(copy); }
    return *this;
}
HostKvLease::HostKvLease(HostKvLease && other) noexcept : allocation_(std::move(other.allocation_)) {}
HostKvLease & HostKvLease::operator=(HostKvLease && other) noexcept {
    if (this != &other) { unpin(allocation_); allocation_ = std::move(other.allocation_); }
    return *this;
}
HostKvLease::~HostKvLease() { unpin(allocation_); }
HostKvSource::HostKvSource(std::shared_ptr<detail::HostKvAllocation> allocation)
    : allocation_(std::move(allocation)) {
    if (allocation_) allocation_->readers.fetch_add(1, std::memory_order_relaxed);
}
HostKvSource::HostKvSource(const HostKvSource & other) : HostKvSource(other.allocation_) {}
HostKvSource & HostKvSource::operator=(const HostKvSource & other) {
    if (this != &other) { HostKvSource copy(other); *this = std::move(copy); }
    return *this;
}
HostKvSource::HostKvSource(HostKvSource && other) noexcept : allocation_(std::move(other.allocation_)) {}
HostKvSource & HostKvSource::operator=(HostKvSource && other) noexcept {
    if (this != &other) { release_source(allocation_); allocation_ = std::move(other.allocation_); }
    return *this;
}
HostKvSource::~HostKvSource() { release_source(allocation_); }
size_t HostKvSource::size() const noexcept { return allocation_ ? allocation_->size : 0; }
HostKvLease HostKvSource::lease() const {
    if (allocation_ && allocation_->account->spill) {
        std::lock_guard<std::mutex> lock(allocation_->mutex);
        return HostKvLease(allocation_);
    }
    return HostKvLease(allocation_);
}
void HostKvSource::read(size_t offset, void * destination, size_t bytes) const {
    lease().read(offset, destination, bytes);
}
const uint8_t * HostKvLease::data() const noexcept {
    return allocation_ ? allocation_->bytes.get() : nullptr;
}
size_t HostKvLease::size() const noexcept { return allocation_ ? allocation_->size : 0; }
void HostKvLease::read(size_t offset, void * destination, size_t bytes) const {
    check_range(size(), offset, destination, bytes);
    if (!bytes) return;
    if (data()) std::memcpy(destination, data() + offset, bytes);
    else allocation_->disk.read(offset, destination, bytes);
}
HostKvRecord::HostKvRecord(std::shared_ptr<detail::HostKvAllocation> allocation)
    : allocation_(std::move(allocation)) {}
size_t HostKvRecord::size() const noexcept { return allocation_ ? allocation_->size : 0; }
HostKvLease HostKvRecord::lease() const {
    if (allocation_ && allocation_->account->spill) {
        std::lock_guard<std::mutex> lock(allocation_->mutex);
        return HostKvLease(allocation_);
    }
    return HostKvLease(allocation_);
}
HostKvSource HostKvRecord::source() const {
    if (allocation_ && allocation_->account->spill) {
        // The record owner serializes source creation with writes. Relocation
        // only changes the tier, so retaining immutable content need not wait
        // for the tier mutex held during a disk write.
        if (!allocation_->sealed.load(std::memory_order_acquire))
            throw MemoryError(MemoryErrorCode::Pending, "deferred KV source must be sealed");
        return HostKvSource(allocation_);
    }
    return HostKvSource(allocation_);
}
const uint8_t * HostKvRecord::data() const {
    if (allocation_ && allocation_->account->spill)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "tiered KV reads require a lease or read()");
    return allocation_ ? allocation_->bytes.get() : nullptr;
}
bool HostKvRecord::mutable_resident() const noexcept {
    return allocation_ && (!allocation_->account->spill ||
        !allocation_->sealed.load(std::memory_order_acquire));
}
uint8_t * HostKvRecord::writable_data() {
    if (!allocation_) return nullptr;
    auto & record = *allocation_;
    auto & account = *record.account;
    if (!account.spill) {
        if (allocation_.use_count() != 1)
            throw MemoryError(MemoryErrorCode::Pending, "native KV record has outstanding read leases");
        return record.bytes.get();
    }
    {
        std::lock_guard<std::mutex> lock(record.mutex);
        if (record.pins.load(std::memory_order_acquire) || record.readers.load(std::memory_order_acquire))
            throw MemoryError(MemoryErrorCode::Pending, "native KV record has outstanding readers");
        // The engine owns a mutable resident buffer already. It cannot be
        // evicted, so a different lane's disk write must not make this borrow
        // wait on the global eviction policy lock.
        if (record.bytes && !record.sealed) return record.bytes.get();
    }
    std::unique_lock<std::mutex> policy(account.policy, std::defer_lock);
    std::unique_lock<std::mutex> lock(record.mutex, std::defer_lock);
    std::lock(policy, lock);
    if (record.pins.load(std::memory_order_acquire) || record.readers.load(std::memory_order_acquire))
        throw MemoryError(MemoryErrorCode::Pending, "native KV record has outstanding read leases");
    if (!record.bytes) {
        account.reserve_locked(policy, record.size, &record);
        try {
            // H is already reserved and this record is exclusively locked.
            // A cold rewrite must not hold the catalog across a physical read;
            // independent lanes may still allocate or relocate other records.
            policy.unlock();
            // The disk read covers the whole block, so it needs no zeroing.
            auto bytes = host_kv_alloc_block(allocation_->account, record.size, false);
            record.disk.read(0, bytes.get(), record.size);
            policy.lock();
            account.register_resident(allocation_);
            record.bytes = std::move(bytes);
            record.resident_size.store(record.size, std::memory_order_release);
        } catch (...) { account.release(record.size); throw; }
    }
    // A writable borrow cannot be evicted until the adapter seals its completed
    // write/DMA. Never invalidate disk before RAM materialization succeeds.
    record.disk.reset();
    record.sealed = false;
    return record.bytes.get();
}
void HostKvRecord::read(size_t offset, void * destination, size_t bytes) const {
    check_range(size(), offset, destination, bytes);
    if (!bytes) return;
    if (allocation_->account->spill) lease().read(offset, destination, bytes);
    else std::memcpy(destination, allocation_->bytes.get() + offset, bytes);
}
void HostKvRecord::write(size_t offset, const void * source, size_t bytes) {
    check_range(size(), offset, source, bytes);
    if (bytes) std::memcpy(writable_data() + offset, source, bytes);
}
void HostKvRecord::seal() {
    if (!allocation_ || !allocation_->account->spill) return;
    // Publication follows the owner's completed write/DMA. Relocation observes
    // this release before reading the bytes; repeated sealing needs no I/O lock.
    allocation_->sealed.store(true, std::memory_order_release);
}
uint64_t HostKvRecord::resident_bytes() const {
    if (!allocation_) return 0;
    if (!allocation_->account->spill) return allocation_->size;
    return allocation_->resident_size.load(std::memory_order_acquire);
}
HostKvReservation::HostKvReservation(std::shared_ptr<detail::HostKvAccount> account, uint64_t bytes)
    : account_(std::move(account)), remaining_(bytes) {}
HostKvReservation::HostKvReservation(HostKvReservation && other) noexcept
    : account_(std::move(other.account_)), remaining_(std::exchange(other.remaining_, 0)) {}
HostKvReservation & HostKvReservation::operator=(HostKvReservation && other) noexcept {
    if (this != &other) {
        release(); account_ = std::move(other.account_);
        remaining_ = std::exchange(other.remaining_, 0);
    }
    return *this;
}
HostKvReservation::~HostKvReservation() { release(); }
void HostKvReservation::release() noexcept {
    if (account_) account_->release(std::exchange(remaining_, 0));
}
namespace {
// Counters behind KVMEM_HOST_PERF=1. Every host KV block is a fresh heap
// allocation today, so an eviction-heavy run pays new/delete per block; these
// numbers are what decides whether an arena is worth its invariants.
struct host_kv_alloc_stats {
    bool     on    = false;
    detail::HostKvArena * arena_view = nullptr;
    uint64_t calls = 0;
    uint64_t bytes = 0;
    uint64_t ns    = 0;
    uint64_t calls_copy = 0, bytes_copy = 0, ns_copy_new = 0;
    uint64_t calls_zero = 0, bytes_zero = 0, ns_zero_new = 0;
    host_kv_alloc_stats() {
        const char * e = std::getenv("KVMEM_HOST_PERF");
        on = e != nullptr && *e != '\0' && *e != '0';
        if (on) std::atexit(report);
    }
    static void report() {
        host_kv_alloc_stats & s = instance();
        std::fprintf(stderr, "[host-alloc] calls=%llu bytes=%llu ms=%.3f\n",
                (unsigned long long) s.calls, (unsigned long long) s.bytes, (double) s.ns / 1e6);
        std::fprintf(stderr, "[host-alloc]   with-src calls=%llu bytes=%llu new_ms=%.3f copy_ms=%.3f\n",
                (unsigned long long) s.calls_copy, (unsigned long long) s.bytes_copy,
                (double) s.ns_copy_new / 1e6,
                (double) (s.ns - s.ns_copy_new - s.ns_zero_new) / 1e6);
        std::fprintf(stderr, "[host-alloc]   zeroed   calls=%llu bytes=%llu new_ms=%.3f\n",
                (unsigned long long) s.calls_zero, (unsigned long long) s.bytes_zero,
                (double) s.ns_zero_new / 1e6);
        if (s.arena_view) {
            std::fprintf(stderr, "[host-alloc]   arena hits=%llu misses=%llu dropped=%llu cached_kb=%zu\n",
                    (unsigned long long) s.arena_view->hits, (unsigned long long) s.arena_view->misses,
                    (unsigned long long) s.arena_view->dropped, s.arena_view->cached / 1024);
        }
        std::fflush(stderr);
    }
    static host_kv_alloc_stats & instance() { static host_kv_alloc_stats s; return s; }
};
} // namespace

HostKvRecord HostKvReservation::allocate(size_t bytes, const void * source) {
    if (!account_ || bytes > remaining_)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "native KV reservation is too small");
    if (!bytes) return {};
    host_kv_alloc_stats & stats = host_kv_alloc_stats::instance();
    stats.arena_view = account_->arena.get();
    const auto t0 = stats.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto allocation = std::make_shared<detail::HostKvAllocation>();
    const auto t_new = stats.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // A parked block still holds the previous tenant's bytes. The old code asked
    // for zeroed storage whenever there was no source, and callers rely on it:
    // the partial-tail path allocates a whole block and then copies only the rows
    // it has, leaving the rest to read as zero. Keep that promise here.
    allocation->bytes = host_kv_alloc_block(account_, bytes, source == nullptr);
    if (stats.on) {
        const auto t_copy = std::chrono::steady_clock::now();
        auto & slot = source ? stats.ns_copy_new : stats.ns_zero_new;
        slot += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_copy - t_new).count();
        (source ? stats.calls_copy : stats.calls_zero) += 1;
        (source ? stats.bytes_copy : stats.bytes_zero) += bytes;
    }
    if (source) std::memcpy(allocation->bytes.get(), source, bytes);
    if (account_->spill) {
        std::lock_guard<std::mutex> policy(account_->policy);
        account_->register_resident(allocation);
    }
    allocation->size = bytes;
    allocation->resident_size.store(bytes, std::memory_order_relaxed);
    allocation->account = account_;
    remaining_ -= bytes;
    if (stats.on) {
        stats.calls += 1;
        stats.bytes += bytes;
        stats.ns    += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
    }
    return HostKvRecord(std::move(allocation));
}
HostKvStorage::HostKvStorage(uint64_t capacity_bytes) : HostKvStorage(capacity_bytes, {}) {}
HostKvStorage::HostKvStorage(uint64_t capacity_bytes, std::shared_ptr<SpillFile> spill)
    : account_(std::make_shared<detail::HostKvAccount>(capacity_bytes, std::move(spill))) {}
HostKvReservation HostKvStorage::reserve(uint64_t bytes) {
    if (account_->spill) {
        std::unique_lock<std::mutex> policy(account_->policy);
        account_->reserve_locked(policy, bytes);
    } else if (!account_->try_charge(bytes)) {
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "native KV RAM capacity exhausted");
    }
    return HostKvReservation(account_, bytes);
}
HostKvRecord HostKvStorage::allocate(size_t bytes, const void * source) {
    return reserve(bytes).allocate(bytes, source);
}
uint64_t HostKvStorage::charged_bytes() const { return account_->usage(); }
uint64_t HostKvStorage::capacity_bytes() const noexcept { return account_->capacity; }
uint64_t HostKvStorage::disk_bytes() const { return account_->spill ? account_->spill->charged_bytes() : 0; }
uint64_t HostKvStorage::disk_capacity_bytes() const noexcept {
    return account_->spill ? account_->spill->capacity_bytes() : 0;
}
SpillIoStats HostKvStorage::io_stats() const noexcept {
    return account_->spill ? account_->spill->stats() : SpillIoStats{};
}
std::shared_ptr<SpillFile> HostKvStorage::spill_file() const noexcept { return account_->spill; }
void HostKvStorage::trim(uint64_t resident_target) {
    if (!account_->spill) return;
    std::unique_lock<std::mutex> policy(account_->policy);
    while (account_->usage() > resident_target && account_->evict_batch(policy, nullptr)) {}
}
} // namespace kvmem
