/*
 * usm_caching_alloc.cpp — production-grade XPU allocator using sycl::malloc_device
 * with a per-device free-list cache.
 *
 * Registered via torch.xpu.memory.XPUPluggableAllocator. Unlike PyTorch's
 * expandable_segments path (zeVirtualMemMap), sycl::malloc_device allocates
 * via zeMemAllocDevice so zeMemGetAllocProperties returns ZE_MEMORY_TYPE_DEVICE
 * and zeMemGetIpcHandle works correctly — fixing both oneCCL incompatibilities:
 *   1. ccl_check_usm_pointers type check
 *   2. Zero-copy IPC path for large tensors
 *
 * Design: segregated free lists, one set per device. Each bucket holds blocks
 * of a specific size class (power-of-2 rounding). Freed blocks are returned to
 * the bucket rather than released to the driver, so subsequent same-size
 * allocations are O(1) without driver overhead. Buckets are protected by a
 * per-device mutex.
 *
 * Build:
 *   icpx -shared -fPIC -fsycl -O2 -o usm_caching_alloc.so usm_caching_alloc.cpp
 *
 * Use (before any XPU initialization):
 *   from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
 *   alloc = XPUPluggableAllocator(
 *       "usm_caching_alloc.so", "xpu_usm_malloc", "xpu_usm_free")
 *   change_current_allocator(alloc)
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Size class bucketing: round up to the nearest power of two, capped at 1 GiB.
// Allocations above the cap are served directly (not pooled).
// ---------------------------------------------------------------------------
static constexpr size_t kBucketCap = size_t(1) << 30;  // 1 GiB

static size_t bucket_size(size_t n) {
    if (n == 0) n = 1;
    if (n > kBucketCap) return n;  // not pooled
    size_t s = size_t(1);
    while (s < n) s <<= 1;
    return s;
}

// ---------------------------------------------------------------------------
// Per-device allocator state
// ---------------------------------------------------------------------------
struct DevicePool {
    std::mutex mtx;
    // bucket_size -> list of free USM pointers of that exact size
    std::unordered_map<size_t, std::vector<void*>> free_lists;
    // ptr -> actual allocated size (for returning to correct bucket on free)
    std::unordered_map<void*, size_t> alloc_sizes;

    void* alloc(size_t requested, sycl::queue* queue) {
        const size_t sz = bucket_size(requested);
        std::lock_guard<std::mutex> lock(mtx);
        if (sz <= kBucketCap) {
            auto& bucket = free_lists[sz];
            if (!bucket.empty()) {
                void* ptr = bucket.back();
                bucket.pop_back();
                return ptr;
            }
        }
        // No cached block — allocate from driver
        void* ptr = sycl::malloc_device(sz, *queue);
        if (ptr) alloc_sizes[ptr] = sz;
        return ptr;
    }

    void free(void* ptr, sycl::queue* queue) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = alloc_sizes.find(ptr);
        if (it == alloc_sizes.end()) {
            // Unknown pointer — shouldn't happen; free it directly
            sycl::free(ptr, *queue);
            return;
        }
        const size_t sz = it->second;
        if (sz <= kBucketCap) {
            free_lists[sz].push_back(ptr);
        } else {
            // Oversized — return to driver immediately
            alloc_sizes.erase(it);
            sycl::free(ptr, *queue);
        }
    }
};

// ---------------------------------------------------------------------------
// Global device table (indexed by device ordinal)
// ---------------------------------------------------------------------------
static constexpr int kMaxDevices = 16;
static DevicePool g_pools[kMaxDevices];

// Debug tracing: set USM_ALLOC_DEBUG=1 to enable
static bool g_debug = (getenv("USM_ALLOC_DEBUG") != nullptr);

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (device < 0 || device >= kMaxDevices) {
        if (g_debug) fprintf(stderr, "[usm_alloc] malloc bad device=%d\n", device);
        return nullptr;
    }
    if (!queue) {
        // queue can be null during early init; fall back to direct alloc
        if (g_debug) fprintf(stderr, "[usm_alloc] malloc null queue, size=%zu dev=%d\n", size, device);
        return nullptr;
    }
    void* ptr = g_pools[device].alloc(size, queue);
    if (g_debug) {
        auto& pool = g_pools[device];
        std::lock_guard<std::mutex> lk(pool.mtx);
        size_t cached = 0;
        for (auto& [sz, v] : pool.free_lists) cached += v.size();
        fprintf(stderr, "[usm_alloc] malloc size=%zu dev=%d -> %p (pool_size=%zu allocs=%zu)\n",
                size, device, ptr, cached, pool.alloc_sizes.size());
    }
    return ptr;
}

void xpu_usm_free(void* ptr, size_t /*size*/, sycl::queue* queue) {
    if (!ptr) return;
    for (int d = 0; d < kMaxDevices; ++d) {
        std::lock_guard<std::mutex> lock(g_pools[d].mtx);
        auto it = g_pools[d].alloc_sizes.find(ptr);
        if (it != g_pools[d].alloc_sizes.end()) {
            const size_t sz = it->second;
            if (sz <= kBucketCap) {
                g_pools[d].free_lists[sz].push_back(ptr);
                if (g_debug)
                    fprintf(stderr, "[usm_alloc] free dev=%d ptr=%p sz=%zu -> POOLED (pool now %zu)\n",
                            d, ptr, sz, g_pools[d].free_lists[sz].size());
            } else {
                g_pools[d].alloc_sizes.erase(it);
                if (queue) sycl::free(ptr, *queue);
                if (g_debug)
                    fprintf(stderr, "[usm_alloc] free dev=%d ptr=%p sz=%zu -> DRIVER (oversized)\n",
                            d, ptr, sz);
            }
            return;
        }
    }
    // Not in any pool: PyTorch may call free for pointers we didn't allocate (e.g. empty tensors)
    if (g_debug) fprintf(stderr, "[usm_alloc] free ptr=%p NOT IN POOL, queue=%p\n", ptr, (void*)queue);
    if (queue) sycl::free(ptr, *queue);
}

// Stats: call from Python via ctypes to query pool state.
// Returns: (num_live_allocs, num_pooled_blocks, total_pooled_bytes)
void xpu_usm_get_stats(size_t* num_live, size_t* num_pooled, size_t* pooled_bytes) {
    *num_live = 0;
    *num_pooled = 0;
    *pooled_bytes = 0;
    for (int d = 0; d < kMaxDevices; ++d) {
        std::lock_guard<std::mutex> lock(g_pools[d].mtx);
        *num_live += g_pools[d].alloc_sizes.size();
        for (auto& [sz, v] : g_pools[d].free_lists) {
            *num_pooled += v.size();
            *pooled_bytes += v.size() * sz;
        }
    }
    // live = total unique allocations (includes both in-use and pooled)
    // pooled = blocks in free list awaiting reuse
    // in-use = live - pooled
}

}  // extern "C"
