/*
 * usm_alloc.cpp — custom XPU allocator using plain sycl::malloc_device (USM).
 *
 * Registered via torch.xpu.memory.XPUPluggableAllocator. Unlike PyTorch's
 * default expandable_segments path (which uses SYCL ext_oneapi_virtual_mem /
 * L0 zeVirtualMemMap), sycl::malloc_device allocates via zeMemAllocDevice so
 * zeMemGetAllocProperties returns ZE_MEMORY_TYPE_DEVICE — accepted by oneCCL.
 *
 * This tests whether replacing the virtual-memory backing with plain USM is
 * sufficient to fix the CCL incompatibility (Option B in the bug report).
 *
 * Build:
 *   icpx -shared -fPIC -fsycl -o usm_alloc.so usm_alloc.cpp
 */
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>

extern "C" {

void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    void* ptr = sycl::malloc_device(size, *queue);
    fprintf(stderr, "[usm_alloc] malloc_device(%zu) on device %d -> %p\n",
            size, device, ptr);
    return ptr;
}

void xpu_usm_free(void* ptr, size_t /*size*/, sycl::queue* queue) {
    sycl::free(ptr, *queue);
}

}  // extern "C"
