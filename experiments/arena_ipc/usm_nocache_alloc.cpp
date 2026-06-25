/*
 * usm_nocache_alloc.cpp — passthrough allocator (no caching, no pooling).
 *
 * Every malloc is sycl::malloc_device; every free is sycl::free.
 * If FSDP+CCL works with this, the bug is in the cache-reuse path of
 * usm_arena_alloc. If it still fails, the bug is in the pluggable-allocator
 * mechanism itself (e.g. PyTorch's deleter callback, queue handling).
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

static bool g_debug = (getenv("USM_ALLOC_DEBUG") != nullptr);

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    if (!queue) return nullptr;
    void* ptr = sycl::malloc_device(size, *queue);
    if (g_debug)
        fprintf(stderr, "[nocache] malloc size=%zu dev=%d -> %p\n", size, device, ptr);
    return ptr;
}

void xpu_usm_free(void* ptr, size_t /*size*/, int device, sycl::queue* queue) {
    if (!ptr || !queue) return;
    sycl::free(ptr, *queue);
    if (g_debug)
        fprintf(stderr, "[nocache] free ptr=%p dev=%d\n", ptr, device);
}

}  // extern "C"
