/*
 * diag_2_l0_ipc_shim.c — LD_PRELOAD shim intercepting Level Zero IPC handle calls.
 *
 * Intercepts zeMemGetIpcHandle and zeMemOpenIpcHandle to log:
 *   - The L0 context handle used
 *   - The pointer being shared
 *   - The return code
 *   - zeMemGetAllocProperties result (memory type + allocation's own context)
 *
 * Build:
 *   gcc -shared -fPIC -o experiments/arena_ipc/diag_2_l0_ipc_shim.so \
 *       experiments/arena_ipc/diag_2_l0_ipc_shim.c -ldl
 *
 * Usage (2-rank CCL reproducer with large tensor — must trigger IPC path):
 *   LD_PRELOAD=$TORCHTUNE/experiments/arena_ipc/diag_2_l0_ipc_shim.so \
 *   XPU_USM_ALLOC_SO=$TORCHTUNE/recipes/dev/usm_arena_alloc.so \
 *   ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0,1 \
 *   torchrun --nproc_per_node=2 \
 *       experiments/arena_ipc/diag_2_repro_large.py
 *
 * Look for lines like:
 *   [L0_IPC] zeMemGetIpcHandle ctx=0x... ptr=0x... -> rc=0 (SUCCESS)
 *   [L0_IPC] zeMemGetIpcHandle ctx=0x... ptr=0x... -> rc=700 (INVALID_ARGUMENT)
 *
 * A non-zero rc from zeMemGetIpcHandle immediately before the GPU page fault
 * confirms this is the failure point.
 *
 * Also intercepts zeMemGetAllocProperties to show what context the allocation
 * belongs to vs. what context CCL queries with — the difference IS the mismatch.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Minimal Level Zero types (avoid including ze_api.h to prevent version conflicts) */
typedef void* ze_context_handle_t;
typedef void* ze_device_handle_t;
typedef void* ze_driver_handle_t;

typedef struct { uint8_t data[64]; } ze_ipc_mem_handle_t;

typedef enum {
    ZE_MEMORY_TYPE_UNKNOWN = 0,
    ZE_MEMORY_TYPE_HOST    = 1,
    ZE_MEMORY_TYPE_DEVICE  = 2,
    ZE_MEMORY_TYPE_SHARED  = 3,
} ze_memory_type_t;

typedef struct {
    uint32_t         stype;
    void*            pNext;
    ze_memory_type_t type;
    uint64_t         id;
    uint64_t         pageSize;
} ze_memory_allocation_properties_t;

#define ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES 0x00000001f
#define ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED 0

static const char* ze_type_str(ze_memory_type_t t) {
    switch(t) {
        case ZE_MEMORY_TYPE_UNKNOWN: return "UNKNOWN(0)";
        case ZE_MEMORY_TYPE_HOST:    return "HOST(1)";
        case ZE_MEMORY_TYPE_DEVICE:  return "DEVICE(2)";
        case ZE_MEMORY_TYPE_SHARED:  return "SHARED(3)";
        default: return "??(>3)";
    }
}

/* Function pointer types */
typedef int (*fn_GetIpcHandle)(ze_context_handle_t, const void*,
                               ze_ipc_mem_handle_t*);
typedef int (*fn_OpenIpcHandle)(ze_context_handle_t, ze_device_handle_t,
                                ze_ipc_mem_handle_t, unsigned int, void**);
typedef int (*fn_GetAllocProps)(ze_context_handle_t, const void*,
                                ze_memory_allocation_properties_t*,
                                ze_device_handle_t*);

static fn_GetIpcHandle   real_GetIpcHandle   = NULL;
static fn_OpenIpcHandle  real_OpenIpcHandle  = NULL;
static fn_GetAllocProps  real_GetAllocProps  = NULL;

static int g_rank = -1;  /* cached MPI rank for log prefix */

static int get_rank(void) {
    if (g_rank >= 0) return g_rank;
    const char* r = getenv("RANK");
    if (!r) r = getenv("LOCAL_RANK");
    g_rank = r ? atoi(r) : 0;
    return g_rank;
}

int zeMemGetIpcHandle(ze_context_handle_t ctx, const void* ptr,
                      ze_ipc_mem_handle_t* handle) {
    if (!real_GetIpcHandle)
        real_GetIpcHandle = (fn_GetIpcHandle)dlsym(RTLD_NEXT, "zeMemGetIpcHandle");

    /* Query alloc properties BEFORE calling the real function */
    ze_memory_type_t mem_type = ZE_MEMORY_TYPE_UNKNOWN;
    if (real_GetAllocProps) {
        ze_memory_allocation_properties_t props;
        memset(&props, 0, sizeof(props));
        props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        ze_device_handle_t dev = NULL;
        real_GetAllocProps(ctx, ptr, &props, &dev);
        mem_type = props.type;
    }

    int rc = real_GetIpcHandle(ctx, ptr, handle);
    fprintf(stderr,
            "[L0_IPC rank%d] zeMemGetIpcHandle(ctx=%p, ptr=%p) "
            "mem_type=%s -> rc=%d %s\n",
            get_rank(), (void*)ctx, ptr,
            ze_type_str(mem_type),
            rc, rc == 0 ? "(SUCCESS)" : "(FAIL)");
    fflush(stderr);
    return rc;
}

int zeMemOpenIpcHandle(ze_context_handle_t ctx, ze_device_handle_t dev,
                       ze_ipc_mem_handle_t handle, unsigned int flags,
                       void** pptr) {
    if (!real_OpenIpcHandle)
        real_OpenIpcHandle = (fn_OpenIpcHandle)dlsym(RTLD_NEXT, "zeMemOpenIpcHandle");

    int rc = real_OpenIpcHandle(ctx, dev, handle, flags, pptr);
    fprintf(stderr,
            "[L0_IPC rank%d] zeMemOpenIpcHandle(ctx=%p, dev=%p, flags=%u) "
            "-> rc=%d %s ptr=%p\n",
            get_rank(), (void*)ctx, (void*)dev, flags,
            rc, rc == 0 ? "(SUCCESS)" : "(FAIL)",
            pptr ? *pptr : NULL);
    fflush(stderr);
    return rc;
}

int zeMemGetAllocProperties(ze_context_handle_t ctx, const void* ptr,
                            ze_memory_allocation_properties_t* props,
                            ze_device_handle_t* dev) {
    if (!real_GetAllocProps)
        real_GetAllocProps = (fn_GetAllocProps)dlsym(RTLD_NEXT, "zeMemGetAllocProperties");

    int rc = real_GetAllocProps(ctx, ptr, props, dev);
    /* Only log if result is unexpected (unknown type or error) to avoid spam */
    if (props && (props->type == ZE_MEMORY_TYPE_UNKNOWN || rc != 0)) {
        fprintf(stderr,
                "[L0_IPC rank%d] zeMemGetAllocProperties(ctx=%p, ptr=%p) "
                "-> rc=%d type=%s\n",
                get_rank(), (void*)ctx, ptr, rc, ze_type_str(props->type));
        fflush(stderr);
    }
    return rc;
}
