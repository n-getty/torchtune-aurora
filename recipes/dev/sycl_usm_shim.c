/*
 * sycl_usm_shim.c — proof-of-concept for the "alternative fix" described in
 * docs/intel_ccl_expandable_segments_bug.md.
 *
 * Intercepts sycl::get_pointer_type() and promotes unknown -> device for
 * pointers created by PyTorch's expandable_segments allocator (via SYCL
 * ext_oneapi_virtual_mem / L0 zeVirtualMemMap), allowing oneCCL collectives
 * to proceed without any oneCCL source changes.
 *
 * Confirmed working: 2-rank allreduce with PYTORCH_ALLOC_CONF=expandable_segments:True
 * succeeds on Intel Max 1550, frameworks/2025.3.1.
 *
 * Build:
 *   gcc -shared -fPIC -o sycl_usm_shim.so sycl_usm_shim.c -ldl
 *
 * Use (testing only — not for production):
 *   LD_PRELOAD=/path/to/sycl_usm_shim.so \
 *   PYTORCH_ALLOC_CONF=expandable_segments:True \
 *   ZE_AFFINITY_MASK=0,1 \
 *   torchrun --nproc_per_node=2 repro_ccl_expandable_segments.py
 *
 * Caveat: promotes ALL unknown-type pointers to device, including genuinely
 * invalid ones. Safe only for testing; a production fix belongs in oneCCL
 * (detect virtual-memory-backed device addresses) or in PyTorch's XPU
 * allocator (register expandable_segments memory as USM device memory).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

/* sycl::usm::alloc enum values (SYCL 2020 spec, stable ABI) */
#define SYCL_USM_DEVICE  1
#define SYCL_USM_UNKNOWN 3

typedef int (*get_pointer_type_fn)(const void *, const void *);
static get_pointer_type_fn real_fn = NULL;

static get_pointer_type_fn resolve_real(void) {
    /* libsycl.so is not in the RTLD_NEXT chain of LD_PRELOAD — it is loaded
     * later. Use RTLD_NOLOAD to get a handle to the already-mapped copy
     * without incrementing its reference count. */
    void *h = dlopen("libsycl.so.8", RTLD_NOLOAD | RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "[sycl_usm_shim] dlopen(libsycl.so.8, NOLOAD) failed: %s\n",
                dlerror());
        return NULL;
    }
    get_pointer_type_fn fn = (get_pointer_type_fn)
        dlsym(h, "_ZN4sycl3_V116get_pointer_typeEPKvRKNS0_7contextE");
    if (!fn)
        fprintf(stderr, "[sycl_usm_shim] dlsym for get_pointer_type failed: %s\n",
                dlerror());
    return fn;
}

/*
 * Export as the mangled name of:
 *   sycl::_V1::get_pointer_type(void const*, sycl::_V1::context const&)
 * oneCCL links against this symbol via PLT from libsycl.so.8; LD_PRELOAD
 * causes our version to be resolved first.
 */
int shim_get_pointer_type(const void *ptr, const void *ctxt)
    __asm__("_ZN4sycl3_V116get_pointer_typeEPKvRKNS0_7contextE");

int shim_get_pointer_type(const void *ptr, const void *ctxt)
{
    if (!real_fn)
        real_fn = resolve_real();

    int result = real_fn(ptr, ctxt);

    if (result == SYCL_USM_UNKNOWN) {
        fprintf(stderr, "[sycl_usm_shim] unknown->device ptr=%p\n", ptr);
        return SYCL_USM_DEVICE;
    }
    return result;
}
