/*
 * usm_pending_alloc.cpp — caching allocator with per-pointer pending-stream list
 *
 * The right design (since pluggable's recordStream() is broken upstream).
 * Each freed pointer goes onto a "pending" list keyed by the queue that
 * freed it. When the same queue requests an allocation, only its own pending
 * pointers are eligible (they're guaranteed in-order on that queue). Cross-
 * queue reuse forces a sycl::queue::wait() on the originating queue first.
 *
 * Behavior is equivalent to recordStream-on-free with the comm queue as the
 * stream. Avoids the unconditional wait() on every free that delayfree pays.
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <deque>

static constexpr size_t SMALL_THRESHOLD = size_t(1) << 20;
static constexpr size_t ALIGNMENT       = 512;
static constexpr int    kMaxDevices     = 16;

static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}
static size_t small_bucket_size(size_t n) {
    if (n == 0) n = 1;
    size_t s = 1;
    while (s < n) s <<= 1;
    return s;
}

struct PendingEntry {
    void*        ptr;
    sycl::queue* freeing_q;
};

struct CachingPool {
    std::mutex                                                mtx;
    // free_lists[size] = pointers ready for reuse (already drained)
    std::unordered_map<size_t, std::vector<void*>>            free_lists;
    // pending[size] = pointers awaiting drain on freeing_q
    std::unordered_map<size_t, std::deque<PendingEntry>>      pending;
    std::unordered_map<void*, size_t>                          alloc_sizes;
    bool                                                       use_power_of_2;

    void* alloc(size_t requested, sycl::queue* q) {
        const size_t sz = use_power_of_2
            ? small_bucket_size(requested)
            : align_up(requested, ALIGNMENT);
        std::lock_guard<std::mutex> lk(mtx);
        auto& bucket = free_lists[sz];
        if (!bucket.empty()) {
            void* ptr = bucket.back();
            bucket.pop_back();
            return ptr;
        }
        // Try to drain a pending entry from the same caller queue without waiting.
        auto& pq = pending[sz];
        for (auto it = pq.begin(); it != pq.end(); ++it) {
            if (it->freeing_q == q) {
                void* ptr = it->ptr;
                pq.erase(it);
                return ptr;
            }
        }
        // Otherwise drain the head of pending (which forces a wait on its queue).
        if (!pq.empty()) {
            auto entry = pq.front();
            pq.pop_front();
            try { entry.freeing_q->wait(); } catch (...) {}
            return entry.ptr;
        }
        void* ptr = sycl::malloc_device(sz, *q);
        if (ptr) alloc_sizes[ptr] = sz;
        return ptr;
    }

    bool try_free(void* ptr, sycl::queue* caller_q) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = alloc_sizes.find(ptr);
        if (it == alloc_sizes.end()) return false;
        pending[it->second].push_back({ptr, caller_q});
        return true;
    }
};

struct DevicePool {
    CachingPool small_pool{.use_power_of_2 = true};
    CachingPool large_pool{.use_power_of_2 = false};
};

static DevicePool g_pools[kMaxDevices];

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (device < 0 || device >= kMaxDevices) return nullptr;
    if (!queue) return nullptr;
    DevicePool& pool = g_pools[device];
    if (size < SMALL_THRESHOLD) return pool.small_pool.alloc(size, queue);
    return pool.large_pool.alloc(size, queue);
}

void xpu_usm_free(void* ptr, size_t /*size*/, int /*device*/, sycl::queue* queue) {
    if (!ptr) return;
    for (int d = 0; d < kMaxDevices; ++d) {
        DevicePool& pool = g_pools[d];
        if (pool.small_pool.try_free(ptr, queue)) return;
        if (pool.large_pool.try_free(ptr, queue)) return;
    }
}

}
