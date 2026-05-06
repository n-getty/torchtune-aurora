# Aurora XPU Allocator Strategy

Memory management on Aurora requires careful interaction between the XPU allocator, the CCL IPC handle cache, and OFI memory registrations. Wrong choices cause either OOM or `banned:1` GPU page faults.

**Bottom line**: use the default XPU allocator with `gc:0.95`. Pluggable allocators cannot work at 32B scale. Safe run length is ~80 steps; use checkpoint-restart.

---

## Production Config (32B multi-node)

```bash
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95
unset XPU_USM_ALLOC_SO
export FI_MR_CACHE_MONITOR=disabled
```

Validated 5/5 steps clean, 40.1–40.3 s/step, Qwen3-32B 2-node XCCL 2-hop weight sync. See `docs/reports/allocator_32b_validation_20260425.md` (archived).

For 3B/4B models: the gen1 caching allocator (`recipes/dev/usm_caching_alloc.so`) works and is validated to 130 steps. **Do not use it at 32B** (see "Why pluggable allocators fail" below).

---

## Key Facts

### 1. Default allocator uses expandable segments (suballocation)

The default XPU caching allocator manages memory as ~20-50 large L0 "segments" from which individual tensors are suballocated. Properties:

- **Stable L0 VAs**: segment VAs are never freed by GC (only suballocations are reclaimed internally). CCL's IPC handle cache holds segment VAs, which remain valid even after GC.
- **GC reclaims suballocations, not segments**: internal `free()` marks a block as available in the pool but doesn't call `sycl::free`. No L0 VA change → no stale IPC handles.
- **GC trigger**: fires only on allocation failure AND `reserved > threshold × total_device_memory`. After step 1, the pool warms up and GC never fires at gc:0.95 (reserved = 59.64 GiB, threshold = 0.95 × 64 = 60.78 GiB — just below).

### 2. FSDP2 uses event-based sync, NOT recordStream

FSDP2 composable API (`torch/distributed/fsdp/_fully_shard/`) uses `wait_stream()` / `wait_event()` exclusively. It never calls `tensor.record_stream()`. Confirmed by grep on Aurora frameworks (2025.3.1):

```bash
# Zero results:
grep -rn 'record_stream' torch/distributed/fsdp/_fully_shard/*.py
# Event sync instead:
grep -n 'wait_stream\|wait_event' torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py
# → lines 273, 361, 531
```

This means the `recordStream` no-op in pluggable allocators (`XPUPluggableAllocator.cpp:79-86`) is irrelevant to FSDP2 cross-stream safety. Pluggable allocator failures are from OFI MR accumulation and L0 IPC handle invalidation, not recordStream races.

### 3. kBucketCap = 8 GiB (not 1 GiB)

The caching allocator pools blocks up to 8 GiB:
```cpp
// recipes/dev/usm_caching_alloc.cpp:59
static constexpr size_t kBucketCap = size_t(8) << 30;  // 8 GiB
```

Critical for XCCL weight sync: the 2-hop broadcast creates ~6 GiB buffers. With kBucketCap = 8 GiB, these are permanently pooled; `sycl::free` is never called, preserving OFI DMA registrations.

### 4. Arena vs caching allocator: IPC sub-allocation bug

Two allocator types; only the caching type is safe at scale:

| Type | Implementation | IPC sub-allocation bug? | Safe at 32B? |
|------|---------------|------------------------|-------------|
| Arena (`usm_arena_alloc.cpp`) | Large slab suballocations | YES — `zeMemGetIpcHandle` returns slab base pointer, not correct sub-offset → GPU page faults | NO |
| Caching (`usm_caching_alloc.cpp`, `.so`) | Per-block `sycl::malloc_device` | NO — each block is a standalone allocation; IPC handle points to correct VA | NO at 32B (see §below); YES at 3B |

The arena allocator's IPC bug: `zeMemGetIpcHandle` on a slab sub-offset returns the slab base address. If another rank uses that handle to access the sub-offset, it reads from VA offset 0 of the slab (wrong data) or causes a GPU page fault. This is why EP AG/RS cannot use the arena allocator.

---

## Why Pluggable Allocators Fail at 32B

Tested with `usm_caching_alloc_v2.so` (gen3, per-block sycl::malloc_device):

**Run 3 (no OOM retry)**: OOM at step 0 optimizer. The gen3 allocator never frees anything. By optimizer time: ~5,330 L0 allocs/tile (1,785 large + 3,545 small), 30.8 GiB in free lists, no VA space for AdamW state. `sycl::malloc_device` returns UR_RESULT_ERROR_OUT_OF_RESOURCES.

**Run 4 (with OOM retry — drain free lists + retry sycl::malloc_device)**: Step 0 complete (optimizer with retry: 48.7s). Step 1 `banned:1` immediately.

Root cause of run 4 failure: the OOM retry called `sycl::free` on cached blocks. CCL's IPC handle cache had registered those VAs from step 0 AllGather operations. Step 1's FSDP2 AllGather tried to reuse a cached IPC handle → L0 found the VA was freed → GPU page fault.

**Fundamental constraint**: any pluggable allocator that creates per-tensor L0 allocs will either:
- Pool everything → OOM (run 3)
- Release under OOM → invalidates CCL IPC handles → banned:1 (run 4)

The only fix would be implementing suballocation within the pluggable allocator — reimplementing PyTorch's CachingAllocator, which already does this correctly.

---

## External Memory Growth

Production 32B runs show ~30 MiB/step growth in `external` (= `l0_used - torch_alloc`), the memory held by CCL/L0 driver outside PyTorch's allocator.

### Source: GRPO-specific, not FSDP2

100-step diagnostic: Qwen3-32B, single-node, 12 tiles FSDP2, **no GRPO, no generation, no weight sync, no optimizer**. External grew only 10 MiB total over 100 steps.

Production GRPO grows ~3,000 MiB over 100 steps (300×). Source is one or more GRPO-specific operations:
1. **XCCL 2-hop weight sync broadcast** — most likely (61 GiB of parameters staged + broadcast each step, creating new CCL internal state)
2. **clip_grad_norm AllReduce** (DTensor-based, separate from FSDP2)
3. XCCL PG lifecycle (new PGs during training)

### Safe run length: ~80 steps

`l0_free` stabilizes at ~2.5 GiB after step 2 (at gc:0.95). External grows ~30 MiB/step → l0_free reaches 0 at ~step 83. Beyond that, CCL behavior is uncertain — it may need fresh L0 allocations that can no longer be satisfied.

**Mitigation**: `save_every_n_steps=20` with checkpoint-restart. At ~40s/step: 20 steps = 13 min training + 5 min restart ≈ 1 step/min effective throughput.

### userfaultfd does NOT prevent banned:1

Tested `FI_MR_CACHE_MONITOR=userfaultfd` cross-node (Slingshot/CXI):

- OFI MR layer: userfaultfd correctly auto-invalidates RDMA registrations when VAs are freed. Zero CXI compatibility issues.
- L0 IPC layer: userfaultfd has no hook into the L0 driver. CCL's IPC handle cache is kernel-space; userfaultfd is user-space.

Crashed at step 8 (after GC at step 7). GC triggered because rank 0's `torch_resv` expanded to 62.54 GiB (userfaultfd tracking overhead increased memory pressure vs 59.64 GiB baseline). GC freed segments → L0 IPC handles stale → banned:1.

**userfaultfd actually makes things worse**: it triggers GC earlier due to additional overhead, without preventing the resulting banned:1. Use `FI_MR_CACHE_MONITOR=disabled`.

### CCL IPC vs OFI MR: two separate caches

| Cache | Owner | Layer | userfaultfd? | CCL_ZE_CACHE threshold? |
|-------|-------|-------|-------------|------------------------|
| OFI MR registrations | libfabric (CXI provider) | User-space | Yes — auto-invalidates freed VAs | No |
| L0 IPC handles | Level Zero driver | Kernel-space | No | Yes — prevents eviction at scale |

`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536` prevents the IPC handle cache from evicting handles that might be reused (default=1000 causes eviction → stale handle access → banned:1 at scale). BUT: 65536 threshold accumulates ~10.85 GiB IPC handle memory by step 1 BWD on 10-tile FSDP2 → OOM at step 2 for single-node 32B.

---

## Allocator Files

| File | Status | Notes |
|------|--------|-------|
| `recipes/dev/usm_caching_alloc.so` | Production for ≤4B | Gen1, power-of-2 bucketing, 130 steps validated |
| `recipes/dev/usm_caching_alloc_v2.cpp` | Research only | Gen3, exact-align + OOM retry; fails at 32B |
| `recipes/dev/usm_arena_alloc.cpp` | Deprecated | Gen2, no cross-stream safety |

---

## Remaining Open Question

The ~30 MiB/step external growth source has not been isolated. The XCCL weight sync broadcast is the leading candidate (61 GiB staged each step). If confirmed, options:

1. **Increase `weight_sync_interval`**: sync every N steps instead of every step. N-step stale weights are acceptable if importance-sampling ratios stay bounded.
2. **Static XCCL buffer** (see `docs/features/vllm_weight_sync.md` §"Potential Next Steps"): reuse fixed VAs for XCCL broadcasts — eliminates VA churn and likely eliminates CCL external growth from this path.
3. **Pin-down test**: run full GRPO with no weight sync for 50 steps and plot external. If external flattens, weight sync is confirmed as the sole source.

---

## Files of Record

| File | Content |
|------|---------|
| `docs/reports/archive/allocator_32b_validation_20260425.md` | Test matrix, step timings, memory profiles for runs 3, 4, gc95, ufd (single-node), CL (cross-node) |
| `docs/reports/archive/allocator_deep_analysis_20260425.md` | FSDP2 recordStream audit, CCL external growth source analysis, alternative approaches |
| `docs/reports/archive/allocator_and_launcher_audit_20260424.md` | Audit findings: kBucketCap=8 GiB (not 1 GiB), launcher status, arena vs caching IPC bug distinction |
| `recipes/dev/usm_caching_alloc.cpp` | Gen1 caching allocator source |
| `recipes/dev/usm_caching_alloc_v2.cpp` | Gen3 allocator with OOM retry (research) |
| `memory/feedback_alloc_conf_env_var.md` | 3B: use usm_caching_alloc.so; 32B: gc:0.8 crashes ~step 28; gc:0.95 validated |
| `memory/feedback_alloc_32b_default.md` | Pluggable allocs OOM/banned:1 at step 0-1; default with gc:0.95 works |
