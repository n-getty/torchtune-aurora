/*
 * usm_arena_alloc.cpp — XPU allocator with coalescing arena
 *
 * Improvement over usm_caching_alloc.cpp: uses a coalescing arena for large
 * allocations (≥ 1 MiB) so adjacent freed blocks can merge and satisfy
 * subsequent requests of any size — eliminating driver-level fragmentation.
 * Small allocations (< 1 MiB) use the same size-class bucket approach as
 * usm_caching_alloc.cpp.
 *
 * All memory is allocated via sycl::malloc_device → ZE_MEMORY_TYPE_DEVICE,
 * fixing both oneCCL incompatibilities with expandable_segments:
 *   1. ccl_check_usm_pointers type check
 *   2. Zero-copy IPC path (zeMemGetIpcHandle) for large tensors
 *
 * Build:
 *   icpx -shared -fPIC -fsycl -O2 -o usm_arena_alloc.so usm_arena_alloc.cpp
 *
 * Use (before any XPU initialization):
 *   from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
 *   alloc = XPUPluggableAllocator("usm_arena_alloc.so", "xpu_usm_malloc", "xpu_usm_free")
 *   change_current_allocator(alloc)
 *
 * Env vars:
 *   USM_ALLOC_DEBUG=1    — enable per-call logging to stderr
 *   USM_ARENA_CHUNK_GB=N — initial arena slab size in GiB (default: 4, range: 1–32)
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static const size_t ARENA_CHUNK_SIZE = []() -> size_t {
    const char* env = getenv("USM_ARENA_CHUNK_GB");
    if (env) {
        long gb = atol(env);
        if (gb >= 1 && gb <= 32) return size_t(gb) << 30;
    }
    return size_t(4) << 30;  // 4 GiB default
}();

static constexpr size_t SMALL_THRESHOLD = size_t(1) << 20;   // 1 MiB: use size-class buckets
static constexpr size_t ALIGNMENT       = 512;                // 512-byte GPU cache line alignment
static constexpr size_t MIN_SPLIT_SIZE  = size_t(128) << 10; // 128 KiB: min usable tail after split
static constexpr size_t BUCKET_CAP      = size_t(1) << 30;   // 1 GiB: small-path direct-alloc threshold
static constexpr int    kMaxDevices     = 16;

static bool g_debug = (getenv("USM_ALLOC_DEBUG") != nullptr);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

static size_t small_bucket_size(size_t n) {
    if (n == 0) n = 1;
    if (n > BUCKET_CAP) return n;  // oversized: direct alloc, not pooled
    size_t s = size_t(1);
    while (s < n) s <<= 1;
    return s;
}

// ---------------------------------------------------------------------------
// Arena data structures
// ---------------------------------------------------------------------------

// Block: a contiguous sub-region within one Arena Chunk.
// Blocks within a chunk are linked in ascending address order.
struct Block {
    uint8_t* ptr;
    size_t   size;
    bool     free;
    Block*   prev;        // lower-address neighbor in same chunk (nullptr = first)
    Block*   next;        // higher-address neighbor in same chunk (nullptr = last)
    int      chunk_idx;   // index into Arena::chunks
};

// Chunk: one sycl::malloc_device allocation that the arena sub-divides.
struct Chunk {
    uint8_t* base;
    size_t   total;
    Block*   head;   // first (lowest-address) block in this chunk
};

// Arena: coalescing allocator backed by one or more Chunks.
struct Arena {
    std::vector<Chunk>                  chunks;
    std::multimap<size_t, Block*>       free_by_size;   // best-fit lookup
    std::unordered_map<void*, Block*>   ptr_to_block;   // live (in-use) blocks
    std::mutex                          mtx;
    sycl::queue*                        last_queue = nullptr;

    size_t total_allocated_bytes = 0;
    size_t total_reserved_bytes  = 0;
    size_t num_coalesces         = 0;
    size_t num_chunk_growths     = 0;
};

// DevicePool: small-path buckets + large-path arena, one per device ordinal.
struct DevicePool {
    std::mutex                                      small_mtx;
    std::unordered_map<size_t, std::vector<void*>> small_free_lists;
    std::unordered_map<void*, size_t>              small_alloc_sizes;
    Arena arena;
};

static DevicePool g_pools[kMaxDevices];

// ---------------------------------------------------------------------------
// Arena internals — all called with arena.mtx already held
// ---------------------------------------------------------------------------

// Remove a specific block from free_by_size. Uses equal_range to handle
// multiple blocks of the same size stored in the multimap.
static void remove_from_free_map(Arena& ar, Block* blk) {
    auto range = ar.free_by_size.equal_range(blk->size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == blk) {
            ar.free_by_size.erase(it);
            return;
        }
    }
    if (g_debug)
        fprintf(stderr, "[usm_arena] BUG: remove_from_free_map: block %p (size=%zu) not found\n",
                (void*)blk, blk->size);
}

// Allocate a new Chunk from the driver. Tries progressively smaller sizes on
// failure (4 GiB → 1 GiB → 256 MiB → 2×min → min). Returns true on success.
static bool grow_arena(Arena& ar, size_t min_size, sycl::queue* q) {
    const size_t min_aligned = align_up(min_size, ALIGNMENT);
    const size_t try_sizes[] = {
        ARENA_CHUNK_SIZE,
        size_t(1) << 30,
        size_t(256) << 20,
        align_up(min_size * 2, ALIGNMENT),
        min_aligned,
    };

    for (size_t target : try_sizes) {
        if (target < min_aligned) continue;
        void* raw = nullptr;
        try { raw = sycl::malloc_device(target, *q); } catch (...) {}
        if (!raw) {
            if (g_debug)
                fprintf(stderr, "[usm_arena] chunk alloc %zu MiB failed, trying smaller\n",
                        target >> 20);
            continue;
        }

        int ci = (int)ar.chunks.size();
        ar.chunks.push_back({(uint8_t*)raw, target, nullptr});
        ar.total_reserved_bytes += target;
        ar.num_chunk_growths++;

        Block* blk   = new Block;
        blk->ptr      = (uint8_t*)raw;
        blk->size     = target;
        blk->free     = true;
        blk->prev     = nullptr;
        blk->next     = nullptr;
        blk->chunk_idx = ci;
        ar.chunks[ci].head = blk;

        ar.free_by_size.emplace(target, blk);
        if (g_debug)
            fprintf(stderr, "[usm_arena] chunk %d: %zu MiB at %p (total chunks=%d)\n",
                    ci, target >> 20, raw, ci + 1);
        return true;
    }
    fprintf(stderr, "[usm_arena] grow_arena FAILED: cannot satisfy %zu bytes\n", min_size);
    return false;
}

// Allocate size bytes from the arena (lock must be held by caller).
static void* arena_malloc_locked(Arena& ar, size_t requested, sycl::queue* q) {
    const size_t size = align_up(requested, ALIGNMENT);

    // Best-fit: smallest free block that is >= size
    auto it = ar.free_by_size.lower_bound(size);
    if (it == ar.free_by_size.end()) {
        if (!grow_arena(ar, size, q)) return nullptr;
        it = ar.free_by_size.lower_bound(size);
        if (it == ar.free_by_size.end()) return nullptr;
    }

    Block* blk = it->second;
    ar.free_by_size.erase(it);

    // Split the tail if large enough to be useful
    const size_t leftover = blk->size - size;
    if (leftover >= MIN_SPLIT_SIZE) {
        Block* tail    = new Block;
        tail->ptr      = blk->ptr + size;
        tail->size     = leftover;
        tail->free     = true;
        tail->prev     = blk;
        tail->next     = blk->next;
        tail->chunk_idx = blk->chunk_idx;
        if (blk->next) blk->next->prev = tail;
        blk->next = tail;
        blk->size = size;
        ar.free_by_size.emplace(leftover, tail);
    }

    blk->free = false;
    ar.ptr_to_block[blk->ptr] = blk;
    ar.total_allocated_bytes  += blk->size;

    if (g_debug)
        fprintf(stderr, "[usm_arena] arena_malloc %zu -> %p (block=%zu free_map=%zu)\n",
                requested, blk->ptr, blk->size, ar.free_by_size.size());
    return blk->ptr;
}

// Return a live block back to the arena, coalescing with free neighbors.
// Lock must be held by caller.
static void arena_free_locked(Arena& ar, void* ptr) {
    auto it = ar.ptr_to_block.find(ptr);
    if (it == ar.ptr_to_block.end()) {
        if (g_debug)
            fprintf(stderr, "[usm_arena] arena_free: %p not in ptr_to_block\n", ptr);
        return;
    }

    Block* blk = it->second;
    if (blk->free) {
        // Double-free: always log, even without USM_ALLOC_DEBUG
        fprintf(stderr, "[usm_arena] ERROR: double-free ptr=%p\n", ptr);
        return;
    }

    ar.ptr_to_block.erase(it);
    ar.total_allocated_bytes -= blk->size;
    blk->free = true;

    // Coalesce with next neighbor (blk absorbs next)
    if (blk->next && blk->next->free) {
        Block* absorb = blk->next;
        remove_from_free_map(ar, absorb);
        blk->size += absorb->size;
        blk->next  = absorb->next;
        if (absorb->next) absorb->next->prev = blk;
        delete absorb;
        ar.num_coalesces++;
    }

    // Coalesce with prev neighbor (prev absorbs blk)
    // prev is already in free_by_size — remove it before resizing.
    if (blk->prev && blk->prev->free) {
        Block* keep   = blk->prev;
        Block* absorb = blk;
        remove_from_free_map(ar, keep);
        keep->size += absorb->size;
        keep->next  = absorb->next;
        if (absorb->next) absorb->next->prev = keep;
        delete absorb;
        blk = keep;
        ar.num_coalesces++;
    }

    ar.free_by_size.emplace(blk->size, blk);
    if (g_debug)
        fprintf(stderr, "[usm_arena] arena_free %p -> block size=%zu coalesces=%zu free_map=%zu\n",
                ptr, blk->size, ar.num_coalesces, ar.free_by_size.size());
}

// ---------------------------------------------------------------------------
// Small path — size-class buckets, identical to usm_caching_alloc.cpp
// ---------------------------------------------------------------------------

static void* small_malloc(DevicePool& pool, size_t requested, sycl::queue* q) {
    const size_t sz = small_bucket_size(requested);
    std::lock_guard<std::mutex> lk(pool.small_mtx);
    if (sz <= BUCKET_CAP) {
        auto& bucket = pool.small_free_lists[sz];
        if (!bucket.empty()) {
            void* ptr = bucket.back();
            bucket.pop_back();
            return ptr;
        }
    }
    void* ptr = sycl::malloc_device(sz, *q);
    if (ptr) pool.small_alloc_sizes[ptr] = sz;
    return ptr;
}

// ---------------------------------------------------------------------------
// Exported API
// ---------------------------------------------------------------------------

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (device < 0 || device >= kMaxDevices) {
        if (g_debug) fprintf(stderr, "[usm_arena] malloc bad device=%d\n", device);
        return nullptr;
    }
    if (!queue) {
        // Null queue during early PyTorch XPU init — cannot allocate yet
        if (g_debug)
            fprintf(stderr, "[usm_arena] malloc null queue size=%zu dev=%d\n", size, device);
        return nullptr;
    }

    DevicePool& pool = g_pools[device];
    void* ptr;

    if (size < SMALL_THRESHOLD) {
        ptr = small_malloc(pool, size, queue);
    } else {
        std::lock_guard<std::mutex> lk(pool.arena.mtx);
        pool.arena.last_queue = queue;
        ptr = arena_malloc_locked(pool.arena, size, queue);
    }

    if (g_debug)
        fprintf(stderr, "[usm_arena] malloc size=%zu dev=%d -> %p (%s)\n",
                size, device, ptr, size < SMALL_THRESHOLD ? "small" : "arena");
    return ptr;
}

void xpu_usm_free(void* ptr, size_t /*size*/, sycl::queue* queue) {
    if (!ptr) return;

    for (int d = 0; d < kMaxDevices; ++d) {
        DevicePool& pool = g_pools[d];

        // Try small path first
        {
            std::lock_guard<std::mutex> lk(pool.small_mtx);
            auto it = pool.small_alloc_sizes.find(ptr);
            if (it != pool.small_alloc_sizes.end()) {
                const size_t sz = it->second;
                if (sz <= BUCKET_CAP) {
                    pool.small_free_lists[sz].push_back(ptr);  // return to pool
                } else {
                    // Oversized small-path block: release to driver
                    pool.small_alloc_sizes.erase(it);
                    if (queue) sycl::free(ptr, *queue);
                }
                if (g_debug)
                    fprintf(stderr, "[usm_arena] free ptr=%p dev=%d small\n", ptr, d);
                return;
            }
        }

        // Try arena path
        {
            std::lock_guard<std::mutex> lk(pool.arena.mtx);
            if (pool.arena.ptr_to_block.count(ptr)) {
                arena_free_locked(pool.arena, ptr);
                if (g_debug)
                    fprintf(stderr, "[usm_arena] free ptr=%p dev=%d arena\n", ptr, d);
                return;
            }
        }
    }

    // Not found in any pool — pass to driver (empty tensors, external allocations).
    // PyTorch occasionally calls free on pointers it didn't allocate via our hook.
    if (g_debug)
        fprintf(stderr, "[usm_arena] free ptr=%p NOT IN POOL queue=%p\n", ptr, (void*)queue);
    if (queue) sycl::free(ptr, *queue);
}

// General stats: (num_live_allocs, num_free_blocks, total_arena_reserved_bytes)
// num_live = live small + live arena blocks
// num_free = pooled small blocks + free arena blocks
// reserved_bytes = total driver memory held by all arena chunks (not counting small-path allocs)
void xpu_usm_get_stats(size_t* num_live, size_t* num_pooled, size_t* reserved_bytes) {
    *num_live      = 0;
    *num_pooled    = 0;
    *reserved_bytes = 0;
    for (int d = 0; d < kMaxDevices; ++d) {
        {
            std::lock_guard<std::mutex> lk(g_pools[d].small_mtx);
            *num_live += g_pools[d].small_alloc_sizes.size();
            for (auto& [sz, v] : g_pools[d].small_free_lists)
                *num_pooled += v.size();
        }
        {
            std::lock_guard<std::mutex> lk(g_pools[d].arena.mtx);
            *num_live      += g_pools[d].arena.ptr_to_block.size();
            *num_pooled    += g_pools[d].arena.free_by_size.size();
            *reserved_bytes += g_pools[d].arena.total_reserved_bytes;
        }
    }
}

// Arena-specific stats for coalescing diagnostics
void xpu_usm_get_arena_stats(size_t* num_growths, size_t* num_coalesces,
                              size_t* reserved_bytes, size_t* allocated_bytes) {
    *num_growths    = 0;
    *num_coalesces  = 0;
    *reserved_bytes = 0;
    *allocated_bytes = 0;
    for (int d = 0; d < kMaxDevices; ++d) {
        std::lock_guard<std::mutex> lk(g_pools[d].arena.mtx);
        *num_growths     += g_pools[d].arena.num_chunk_growths;
        *num_coalesces   += g_pools[d].arena.num_coalesces;
        *reserved_bytes  += g_pools[d].arena.total_reserved_bytes;
        *allocated_bytes += g_pools[d].arena.total_allocated_bytes;
    }
}

}  // extern "C"
