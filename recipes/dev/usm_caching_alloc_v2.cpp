/*
 * usm_caching_alloc_v2.cpp — Gen3 XPU caching allocator
 *
 * Combines two proven designs:
 *   - usm_arena_alloc.cpp (gen2): exact-alignment for large allocs, eliminating
 *     power-of-2 waste that OOMed 32B at step 4
 *   - usm_caching_alloc.cpp (gen1): alloc-time queue->wait() for cross-stream
 *     safety, covering the XPUPluggableAllocator recordStream no-op
 *
 * OOM RETRY:
 *   When sycl::malloc_device fails, cached blocks on the same device are freed
 *   back to L0 and the allocation is retried. This handles the optimizer state
 *   creation spike (step 0) where forward/backward activation caches consume
 *   all available L0 memory before AdamW exp_avg/exp_avg_sq are allocated.
 *
 * Build:
 *   icpx -shared -fPIC -fsycl -O2 -o usm_caching_alloc_v2.so usm_caching_alloc_v2.cpp
 *
 * Env vars:
 *   USM_ALLOC_DEBUG=1  — key events only (OOM, release, periodic stats)
 *   USM_ALLOC_DEBUG=2  — verbose per-call logging to stderr
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
static constexpr size_t STATS_INTERVAL  = 10000;

static int g_debug = 0;

static void init_debug() __attribute__((constructor));
static void init_debug() {
    const char* v = getenv("USM_ALLOC_DEBUG");
    if (v) {
        g_debug = atoi(v);
        if (g_debug == 0 && v[0] != '0') g_debug = 1;
    }
}

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

    size_t num_driver_allocs = 0;
    size_t num_cache_hits    = 0;
    size_t num_ops           = 0;

    void* try_alloc_device(size_t sz, sycl::queue* q) {
        void* ptr = nullptr;
        try { ptr = sycl::malloc_device(sz, *q); } catch (...) { ptr = nullptr; }
        return ptr;
    }

    size_t release_cached(sycl::queue* q) {
        // Drain all pending work before freeing — covers cross-stream comm ops
        if (q) { try { q->wait(); } catch (...) {} }
        std::lock_guard<std::mutex> lk(mtx);
        size_t freed_bytes = 0;
        size_t freed_count = 0;
        for (auto& [sz, bucket] : free_lists) {
            for (void* ptr : bucket) {
                try { sycl::free(ptr, *q); } catch (...) {}
                alloc_sizes.erase(ptr);
                freed_bytes += sz;
                freed_count++;
            }
            bucket.clear();
        }
        if (g_debug >= 1 && freed_count > 0)
            fprintf(stderr, "[usm_alloc_v2] RELEASE %s: freed %zu blocks (%.1f MiB)\n",
                    use_power_of_2 ? "small" : "large",
                    freed_count, freed_bytes / (1024.0 * 1024.0));
        return freed_bytes;
    }

    void* alloc(size_t requested, sycl::queue* q) {
        const size_t sz = use_power_of_2
            ? small_bucket_size(requested)
            : align_up(requested, ALIGNMENT);

        void* cached = nullptr;
        {
            std::lock_guard<std::mutex> lk(mtx);
            auto& bucket = free_lists[sz];
            if (!bucket.empty()) {
                cached = bucket.back();
                bucket.pop_back();
                num_cache_hits++;
            }
        }

        if (cached) {
            if (q) { try { q->wait(); } catch (...) {} }

            if (g_debug >= 2)
                fprintf(stderr, "[usm_alloc_v2] alloc size=%zu -> %p CACHED (hits=%zu)\n",
                        requested, cached, num_cache_hits);
            return cached;
        }

        void* ptr = try_alloc_device(sz, q);
        if (ptr) {
            std::lock_guard<std::mutex> lk(mtx);
            alloc_sizes[ptr] = sz;
            num_driver_allocs++;
            if (g_debug >= 2)
                fprintf(stderr, "[usm_alloc_v2] alloc size=%zu -> %p NEW (total=%zu)\n",
                        requested, ptr, num_driver_allocs);
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
    CachingPool large_pool{.use_power_of_2 = false};  // >= 1 MiB
};

static DevicePool g_pools[kMaxDevices];

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (device < 0 || device >= kMaxDevices) return nullptr;
    if (!queue) return nullptr;

    DevicePool& pool = g_pools[device];
    auto& target = (size < SMALL_THRESHOLD) ? pool.small_pool : pool.large_pool;

    void* ptr = target.alloc(size, queue);
    if (ptr) return ptr;

    // OOM — release cached blocks from BOTH pools on this device, then retry
    if (g_debug >= 1)
        fprintf(stderr, "[usm_alloc_v2] OOM on dev %d for %zu bytes — releasing caches\n",
                device, size);
    size_t freed = pool.small_pool.release_cached(queue);
    freed += pool.large_pool.release_cached(queue);
    if (g_debug >= 1)
        fprintf(stderr, "[usm_alloc_v2] Released %.1f MiB total, retrying\n",
                freed / (1024.0 * 1024.0));

    ptr = target.alloc(size, queue);
    if (!ptr && g_debug >= 1)
        fprintf(stderr, "[usm_alloc_v2] RETRY FAILED on dev %d for %zu bytes\n",
                device, size);
    return ptr;
}

// 4-param free signature required by XPUPluggableAllocator framework.
void xpu_usm_free(void* ptr, size_t /*size*/, int device, sycl::queue* /*queue*/) {
    if (!ptr) return;

    // Fast path: try the specified device first
    if (device >= 0 && device < kMaxDevices) {
        DevicePool& pool = g_pools[device];
        if (pool.small_pool.try_free(ptr)) {
            if (g_debug >= 2)
                fprintf(stderr, "[usm_alloc_v2] free ptr=%p dev=%d small\n", ptr, device);
            return;
        }
        if (pool.large_pool.try_free(ptr)) {
            if (g_debug >= 2)
                fprintf(stderr, "[usm_alloc_v2] free ptr=%p dev=%d large\n", ptr, device);
            return;
        }
    }

    // Fallback: scan all devices
    for (int d = 0; d < kMaxDevices; ++d) {
        if (d == device) continue;
        DevicePool& pool = g_pools[d];
        if (pool.small_pool.try_free(ptr)) {
            if (g_debug >= 2)
                fprintf(stderr, "[usm_alloc_v2] free ptr=%p dev=%d small\n", ptr, d);
            return;
        }
        if (pool.large_pool.try_free(ptr)) {
            if (g_debug >= 2)
                fprintf(stderr, "[usm_alloc_v2] free ptr=%p dev=%d large\n", ptr, d);
            return;
        }
    }

    if (g_debug >= 1)
        fprintf(stderr, "[usm_alloc_v2] free ptr=%p NOT IN POOL\n", ptr);
}

void xpu_usm_get_stats(size_t* num_live, size_t* num_pooled, size_t* reserved_bytes) {
    *num_live = 0;
    *num_pooled = 0;
    *reserved_bytes = 0;
    for (int d = 0; d < kMaxDevices; ++d) {
        {
            std::lock_guard<std::mutex> lk(g_pools[d].small_pool.mtx);
            *num_live += g_pools[d].small_pool.alloc_sizes.size();
            for (auto& [sz, v] : g_pools[d].small_pool.free_lists) {
                *num_pooled += v.size();
                *reserved_bytes += v.size() * sz;
            }
        }
        {
            std::lock_guard<std::mutex> lk(g_pools[d].large_pool.mtx);
            *num_live += g_pools[d].large_pool.alloc_sizes.size();
            for (auto& [sz, v] : g_pools[d].large_pool.free_lists) {
                *num_pooled += v.size();
                *reserved_bytes += v.size() * sz;
            }
        }
    }
}

}  // extern "C"
