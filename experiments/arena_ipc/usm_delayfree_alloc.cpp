/*
 * usm_delayfree_alloc.cpp — caching allocator that DEFERS recycling
 *
 * Hypothesis test: the pluggable XPUPluggableAllocator's recordStream() is a
 * no-op (set_record_stream_fn never wired by the Python wrapper). FSDP2 thus
 * frees comm-stream-pending buffers, which the cache then immediately reuses,
 * and the still-pending kernel reads a recycled virtual address → page fault.
 *
 * If we delay returning a freed pointer to the cache by an idempotent
 * synchronization (sycl::queue::wait on the queue passed to free), the
 * pending kernel will have completed before the address is reused.
 *
 * If this allocator PASSES the FSDP test where arena+caching FAILS, we have
 * confirmed the recordStream-missing root cause.
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

static constexpr size_t SMALL_THRESHOLD = size_t(1) << 20;
static constexpr size_t ALIGNMENT       = 512;
static constexpr int    kMaxDevices     = 16;

static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

static size_t small_bucket_size(size_t n) {
    if (n == 0) n = 1;
    size_t s = size_t(1);
    while (s < n) s <<= 1;
    return s;
}

struct CachingPool {
    std::mutex                                      mtx;
    std::unordered_map<size_t, std::vector<void*>> free_lists;
    std::unordered_map<void*, size_t>              alloc_sizes;
    bool                                            use_power_of_2;

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
        void* ptr = sycl::malloc_device(sz, *q);
        if (ptr) alloc_sizes[ptr] = sz;
        return ptr;
    }

    bool try_free(void* ptr, sycl::queue* caller_q) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = alloc_sizes.find(ptr);
        if (it == alloc_sizes.end()) return false;
        // KEY DIFFERENCE: drain the queue before recycling the pointer, so any
        // pending kernel from this stream finishes before the address is reused.
        if (caller_q) {
            try { caller_q->wait(); } catch (...) {}
        }
        free_lists[it->second].push_back(ptr);
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
    if (size < SMALL_THRESHOLD) {
        return pool.small_pool.alloc(size, queue);
    } else {
        return pool.large_pool.alloc(size, queue);
    }
}

void xpu_usm_free(void* ptr, size_t /*size*/, int /*device*/, sycl::queue* queue) {
    if (!ptr) return;
    for (int d = 0; d < kMaxDevices; ++d) {
        DevicePool& pool = g_pools[d];
        if (pool.small_pool.try_free(ptr, queue)) return;
        if (pool.large_pool.try_free(ptr, queue)) return;
    }
}

}  // extern "C"
