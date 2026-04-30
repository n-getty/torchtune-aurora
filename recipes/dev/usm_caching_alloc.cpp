/*
 * usm_caching_alloc.cpp — XPU caching allocator using sycl::malloc_device with
 * per-device power-of-2 free-list buckets.
 *
 * Registered via torch.xpu.memory.XPUPluggableAllocator. Unlike PyTorch's
 * expandable_segments path (zeVirtualMemMap), sycl::malloc_device allocates
 * via zeMemAllocDevice so zeMemGetAllocProperties returns ZE_MEMORY_TYPE_DEVICE
 * and oneCCL ccl_check_usm_pointers passes.
 *
 * WHY A CUSTOM ALLOCATOR (not expandable_segments:False + default allocator):
 *   The default XPU allocator allocates a fresh VA each step for the FSDP
 *   AllGather output. Each new VA triggers a new OFI/libfabric DMA registration
 *   (FI_MR_CACHE_MONITOR=disabled prevents automatic deregistration). Over N
 *   steps, N × 6 GiB registrations accumulate → L0 free → 0, 50s GC stall,
 *   banned:1 crash. This allocator pools the AllGather buffer at a stable VA so
 *   OFI registers it once and reuses the registration every step.
 *
 * CROSS-STREAM SAFETY — wait at ALLOC time, not free time:
 *   XPUPluggableAllocator.recordStream() is a no-op in PyTorch's Python wrapper
 *   (set_record_stream_fn is never wired). Waiting at FREE time (compute stream)
 *   doesn't help because the ReduceScatter runs on the comm stream. Instead, we
 *   wait for the REQUESTING queue at ALLOC time. FSDP2 guarantees that before the
 *   compute queue allocates the next AllGather buffer, it has already issued a
 *   stream-event wait for the comm stream's RS completion. So compute_queue.wait()
 *   at alloc time covers the comm stream sync implicitly, with near-zero overhead
 *   (the queue is already idle by the time alloc is called).
 *
 * Design: segregated free lists, one set per device. Each bucket holds blocks
 * of a specific power-of-2 size class. Freed blocks return immediately to the
 * bucket. Alloc waits for the requesting queue before returning a pooled block.
 * Buckets are protected by a per-device mutex.
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
// Size class bucketing: round up to the nearest power of two, capped at 8 GiB.
// Allocations above the cap are served directly (not pooled).
//
// 8 GiB covers the full top-level FSDP AllGather output (gene3b = 6 GiB,
// rounds to 8 GiB). Power-of-2 bucketing makes variable-sized logprob tensors
// reuse the same bucket across steps → no OOM accumulation.
// ---------------------------------------------------------------------------
static constexpr size_t kBucketCap = size_t(8) << 30;  // 8 GiB

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

    // Returns a pooled block (with ptr removed from free list) and the queue
    // to wait for, or nullptr if no block available.
    void* try_pop(size_t sz) {
        if (sz > kBucketCap) return nullptr;
        std::lock_guard<std::mutex> lock(mtx);
        auto& bucket = free_lists[sz];
        if (bucket.empty()) return nullptr;
        void* ptr = bucket.back();
        bucket.pop_back();
        return ptr;
    }

    void* alloc_new(size_t sz, sycl::queue* queue) {
        void* ptr = sycl::malloc_device(sz, *queue);
        if (ptr) {
            std::lock_guard<std::mutex> lock(mtx);
            alloc_sizes[ptr] = sz;
        }
        return ptr;
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

// Signature: (size, device, queue*) — framework calls with these 3 args.
void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (device < 0 || device >= kMaxDevices) {
        if (g_debug) fprintf(stderr, "[usm_alloc] malloc bad device=%d\n", device);
        return nullptr;
    }
    if (!queue) {
        if (g_debug) fprintf(stderr, "[usm_alloc] malloc null queue size=%zu dev=%d\n", size, device);
        return nullptr;
    }

    const size_t sz = bucket_size(size);
    void* ptr = g_pools[device].try_pop(sz);

    if (ptr) {
        // Pooled reuse: wait for the requesting (compute) queue BEFORE returning.
        // FSDP2 guarantees that before the compute queue allocates the next AllGather
        // buffer, it has issued a stream-event wait for the RS comm stream. So
        // queue->wait() here transitively ensures RS completion, covering the
        // recordStream no-op bug in XPUPluggableAllocator.
        try { queue->wait(); } catch (...) {}
        if (g_debug) {
            auto& pool = g_pools[device];
            std::lock_guard<std::mutex> lk(pool.mtx);
            size_t cached = 0;
            for (auto& [s, v] : pool.free_lists) cached += v.size();
            fprintf(stderr, "[usm_alloc] malloc size=%zu dev=%d -> %p REUSED (pool=%zu)\n",
                    size, device, ptr, cached);
        }
        return ptr;
    }

    // No cached block — allocate from driver
    ptr = g_pools[device].alloc_new(sz, queue);
    if (g_debug) {
        auto& pool = g_pools[device];
        std::lock_guard<std::mutex> lk(pool.mtx);
        size_t cached = 0;
        for (auto& [s, v] : pool.free_lists) cached += v.size();
        fprintf(stderr, "[usm_alloc] malloc size=%zu dev=%d -> %p NEW (allocs=%zu)\n",
                size, device, ptr, pool.alloc_sizes.size());
    }
    return ptr;
}

// Signature: (ptr, size, device, queue*) — 4-param version required by framework.
// Note: 3-param version (missing int device) causes SIGSEGV — device int is read
// as the queue pointer on the next stack slot.
//
// For pooled blocks: return to pool immediately (no wait). The wait happens at
// the next alloc() call, which is safe because FSDP2 syncs compute←comm before
// the next AllGather.
void xpu_usm_free(void* ptr, size_t /*size*/, int /*device*/, sycl::queue* queue) {
    if (!ptr) return;
    for (int d = 0; d < kMaxDevices; ++d) {
        size_t sz = 0;
        bool found = false;
        bool oversized = false;
        {
            std::lock_guard<std::mutex> lock(g_pools[d].mtx);
            auto it = g_pools[d].alloc_sizes.find(ptr);
            if (it == g_pools[d].alloc_sizes.end()) continue;
            sz = it->second;
            found = true;
            oversized = (sz > kBucketCap);
            if (!oversized) {
                // Pooled: push back immediately, no wait here.
                // Safety covered by queue->wait() in xpu_usm_malloc() above.
                g_pools[d].free_lists[sz].push_back(ptr);
                if (g_debug)
                    fprintf(stderr, "[usm_alloc] free dev=%d ptr=%p sz=%zu -> POOLED (pool now %zu)\n",
                            d, ptr, sz, g_pools[d].free_lists[sz].size());
            } else {
                g_pools[d].alloc_sizes.erase(it);
                if (g_debug)
                    fprintf(stderr, "[usm_alloc] free dev=%d ptr=%p sz=%zu -> DRIVER\n", d, ptr, sz);
            }
        }
        if (found && oversized) {
            // Oversized: free to driver. Wait first to avoid use-after-free.
            if (queue) { try { queue->wait(); } catch (...) {} }
            if (queue) sycl::free(ptr, *queue);
        }
        return;
    }
    // Not in any pool (e.g. empty tensors PyTorch allocated before change_current_allocator)
    if (g_debug) fprintf(stderr, "[usm_alloc] free ptr=%p NOT IN POOL queue=%p\n", ptr, (void*)queue);
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
