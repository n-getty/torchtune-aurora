# Audit Report: Caching Allocator & 32B Production Launcher
**Date**: 2026-04-24

This report documents an audit of the recent project status updates (`status.md`, `32b_grpo_status_20260424.md`, `moe_integration.md`) against the actual state of the codebase and launcher scripts. Three key contradictions were identified that impact the stability and planning of the 32B GRPO production runs.

---

## Finding 1: Caching Allocator Pooling Limit (`kBucketCap`)

**The Contradiction:**
* `docs/status.md` (lines 534-535) states: "Blocks ≤ 1 GiB are permanently pooled (never freed to L0 driver)... Large blocks > 1 GiB are freed to L0 immediately after use".
* `docs/reports/32b_grpo_status_20260424.md` (lines 125-127) states: "The XCCL weight sync creates ~6 GiB buffers per tile. `bucket_size(6 GiB)` rounds up to 8 GiB = exactly `kBucketCap`. This buffer is pooled, never `sycl::free`'d".

**The Fact (Source Code):**
In `recipes/dev/usm_caching_alloc.cpp` (line 59), the cap is explicitly defined as 8 GiB:
```cpp
static constexpr size_t kBucketCap = size_t(8) << 30;  // 8 GiB
```

**Implication:**
The `status.md` claim is incorrect. The 8 GiB cap is critical because the XCCL weight sync requires ~6 GiB buffers. If the cap were truly 1 GiB, the allocator would call `sycl::free` on the weight sync buffers, destroying their OFI DMA registrations and causing the late-step GPU page fault (`banned:1`) the allocator is designed to prevent.

---

## Finding 2: 32B Production Launcher Status

**The Contradiction:**
* `docs/status.md` (lines 541-542) claims that both launchers (`run_3b_gene_recall_production.sh` and `run_32b_2hop_production.sh`) have been updated to use the `usm_caching_alloc.so` pluggable allocator.

**The Fact (Source Code):**
While the 3B launcher was correctly updated, `experiments/multinode_32b/run_32b_2hop_production.sh` remains outdated. It currently lacks the `XPU_USM_ALLOC_SO` variable and still relies on the default allocator configuration:
```bash
# Line 254 in run_32b_2hop_production.sh
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
```

**Implication:**
If launched in its current state, the 32B production job will crash at step 29 due to the default allocator triggering a garbage collection on the XCCL weight sync buffers (as documented under "Crash C" in `32b_grpo_status_20260424.md`).

---

## Finding 3: Caching Allocator & IPC Sub-allocation Bug Equivalency

**The Contradiction:**
* `docs/features/moe_integration.md` (lines 628-629) states: "**Do not** retry XCCL for AG/RS with the arena/caching USM allocator — the IPC sub-allocation bug bites at 26B+ scale."

**The Fact (Source Code):**
This is a false equivalence. The IPC sub-allocation bug occurs when an allocator sub-allocates from large memory slabs (which the *arena* allocator does), causing `zeMemGetIpcHandle` to return handles for the slab base instead of the correct sub-offset. 
The *caching* allocator (`usm_caching_alloc.cpp`, lines 91-98) does not use slabs. It calls `sycl::malloc_device` directly for each pooled block, returning the base pointer. 

**Implication:**
The warning in `moe_integration.md` is incorrect regarding the caching allocator. The caching allocator safely bypasses the IPC sub-allocation bug and remains a viable, high-priority solution for 32B scale runs.

---

## Recommended Actions for Validation

Before spending compute resources on the 32B production run, we recommend independently validating these findings and making the following changes:

1. **Update `run_32b_2hop_production.sh`**:
   - Remove `export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6`.
   - Add `export XPU_USM_ALLOC_SO="${TT_DIR}/recipes/dev/usm_caching_alloc.so"`.
2. **Revise `status.md`**:
   - Correct the pooling limit description (8 GiB, not 1 GiB).
   - Retract the claim that the 32B launcher was already updated.
3. **Revise `moe_integration.md`**:
   - Clarify that the caching allocator does not suffer from the IPC sub-allocation bug, removing it from the warning.
