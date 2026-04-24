/*
 * usm_arena_alloc.cpp — XPU caching allocator with exact-size pooling
 *
 * Two allocation tiers:
 *
 *   < 1 MiB (SMALL_THRESHOLD):
 *       Power-of-2 size-class buckets with direct sycl::malloc_device.
 *       Fast path for small scratch tensors. Rounding waste is negligible
 *       at small sizes and improves cache hit rate.
 *
 *   >= 1 MiB:
 *       Exact-aligned direct sycl::malloc_device with caching.
 *       Each allocation is a first-class USM pointer — zeMemGetIpcHandle
 *       returns a valid handle, so CCL's IPC zero-copy path works.
 *       Exact alignment (512-byte) instead of power-of-2 rounding
 *       eliminates memory waste (e.g., 1.0 MiB stays 1.0 MiB, not 2.0 MiB).
 *       FSDP uses consistent sizes each step, so cache hit rate is near-perfect.
 *
 * Why no arena/sub-allocation? CCL's intra-node IPC path calls
 * zeMemGetIpcHandle on every GPU-to-GPU transfer, regardless of size or
 * algorithm selection. Sub-allocated pointers (base + offset from a slab)
 * cause GPU page faults at 0xffffff8000000000. This is a Level Zero / oneCCL
 * constraint on Aurora — algorithm overrides (naive/direct) do NOT bypass it.
 *
 * Why not power-of-2 rounding? The gen1 allocator (usm_caching_alloc.so)
 * used power-of-2 buckets but caused OOM at step 4 for 32B models.
 * Example: 1054720 bytes → 2 MiB (99% waste), 40 MiB → 64 MiB (60% waste).
 * With exact alignment: 1054720 → 1055232 (0.05% waste).
 *
 * Build:
 *   icpx -shared -fPIC -fsycl -O2 -o usm_arena_alloc.so usm_arena_alloc.cpp
 *
 * Env vars:
 *   USM_ALLOC_DEBUG=1  — enable per-call logging to stderr
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

static constexpr size_t SMALL_THRESHOLD = size_t(1) << 20;  // 1 MiB
static constexpr size_t ALIGNMENT       = 512;
static constexpr int    kMaxDevices     = 16;

static bool g_debug = (getenv("USM_ALLOC_DEBUG") != nullptr);

static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

static size_t small_bucket_size(size_t n) {
    if (n == 0) n = 1;
    size_t s = size_t(1);
    while (s < n) s <<= 1;
    return s;
}

// CachingPool: direct sycl::malloc_device with free-list caching.
// Keyed by allocation size for recycling.
struct CachingPool {
    std::mutex                                      mtx;
    std::unordered_map<size_t, std::vector<void*>> free_lists;
    std::unordered_map<void*, size_t>              alloc_sizes;
    bool                                            use_power_of_2;

    size_t num_driver_allocs = 0;
    size_t num_cache_hits    = 0;

    void* alloc(size_t requested, sycl::queue* q) {
        const size_t sz = use_power_of_2
            ? small_bucket_size(requested)
            : align_up(requested, ALIGNMENT);
        std::lock_guard<std::mutex> lk(mtx);
        auto& bucket = free_lists[sz];
        if (!bucket.empty()) {
            void* ptr = bucket.back();
            bucket.pop_back();
            num_cache_hits++;
            return ptr;
        }
        void* ptr = sycl::malloc_device(sz, *q);
        if (ptr) {
            alloc_sizes[ptr] = sz;
            num_driver_allocs++;
        }
        return ptr;
    }

    bool try_free(void* ptr) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = alloc_sizes.find(ptr);
        if (it == alloc_sizes.end()) return false;
        free_lists[it->second].push_back(ptr);
        return true;
    }
};

struct DevicePool {
    CachingPool small_pool{.use_power_of_2 = true};   // < 1 MiB
    CachingPool large_pool{.use_power_of_2 = false};   // >= 1 MiB
};

static DevicePool g_pools[kMaxDevices];

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (device < 0 || device >= kMaxDevices) return nullptr;
    if (!queue) return nullptr;

    DevicePool& pool = g_pools[device];
    void* ptr;
    const char* path;

    if (size < SMALL_THRESHOLD) {
        ptr = pool.small_pool.alloc(size, queue);
        path = "small";
    } else {
        ptr = pool.large_pool.alloc(size, queue);
        path = "large";
    }

    if (g_debug)
        fprintf(stderr, "[usm_alloc] malloc size=%zu dev=%d -> %p (%s)\n",
                size, device, ptr, path);
    return ptr;
}

void xpu_usm_free(void* ptr, size_t /*size*/, int /*device*/, sycl::queue* /*queue*/) {
    if (!ptr) return;

    for (int d = 0; d < kMaxDevices; ++d) {
        DevicePool& pool = g_pools[d];
        if (pool.small_pool.try_free(ptr)) {
            if (g_debug)
                fprintf(stderr, "[usm_alloc] free ptr=%p dev=%d small\n", ptr, d);
            return;
        }
        if (pool.large_pool.try_free(ptr)) {
            if (g_debug)
                fprintf(stderr, "[usm_alloc] free ptr=%p dev=%d large\n", ptr, d);
            return;
        }
    }

    if (g_debug)
        fprintf(stderr, "[usm_alloc] free ptr=%p NOT IN POOL\n", ptr);
}

void xpu_usm_get_stats(size_t* num_live, size_t* num_pooled, size_t* reserved_bytes) {
    *num_live = 0;
    *num_pooled = 0;
    *reserved_bytes = 0;
    for (int d = 0; d < kMaxDevices; ++d) {
        {
            std::lock_guard<std::mutex> lk(g_pools[d].small_pool.mtx);
            *num_live += g_pools[d].small_pool.alloc_sizes.size();
            for (auto& [sz, v] : g_pools[d].small_pool.free_lists)
                *num_pooled += v.size();
        }
        {
            std::lock_guard<std::mutex> lk(g_pools[d].large_pool.mtx);
            *num_live += g_pools[d].large_pool.alloc_sizes.size();
            for (auto& [sz, v] : g_pools[d].large_pool.free_lists)
                *num_pooled += v.size();
        }
    }
}

}  // extern "C"
