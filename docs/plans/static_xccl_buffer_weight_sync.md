# Static XCCL Buffers for Weight Sync

## Context and Rationale

During the 32B GRPO runs using 2-hop XCCL weight sync, we observed a steady external memory growth of ~30 MiB per step. Over 80 steps, this depletes the available L0 free memory (which stabilizes around ~2.5 GiB), forcing a manual checkpoint and restart.

An isolated FSDP2 diagnostic run with static sequence lengths and no weight sync showed **zero** external memory growth (10 MiB over 100 steps). This proved that the memory leak is not an inherent flaw in FSDP2 base collectives, but is instead localized to GRPO-specific operations—primarily the 61 GiB XCCL broadcast during weight sync.

### The Mechanism of the Leak
1. **Allocator Fragmentation**: During the GRPO generation phase, vLLM produces activations of variable sequence lengths. This leads to heavy fragmentation in PyTorch's `expandable_segments` caching allocator.
2. **Dynamic Buffer Allocation**: In the `_bg_xccl_broadcast` thread (training side) and `receive_weights_xccl_streaming` (vLLM side), the code dynamically allocates buffers (e.g., `gpu_temp = torch.empty(max_numel)` and `recv_buf = torch.empty(batch_numel)`) for each chunk of weights (~1 GiB).
3. **Virtual Address Churn**: Due to the fragmentation caused by generation, the caching allocator often cannot find a perfectly contiguous 1 GiB block in its free list. It asks the L0 driver for a new memory segment, resulting in a **new Virtual Address (VA)** for the broadcast buffer.
4. **CCL IPC Handle Accumulation**: When PyTorch passes this new VA to `_xccl_wsync_pg.broadcast()`, oneCCL creates a new Memory Region (MR) and IPC handle. Because we use `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536` to prevent step-2 evictions, CCL never evicts these handles. Since the allocator caches the VAs rather than freeing them to the driver, CCL accumulates an MR entry for every single 1 GiB segment the allocator ever touches. 

This results in the ~30 MiB/step external growth.

### The Proposed Solution
To eliminate the Virtual Address churn entirely, we will replace the dynamic buffer allocations with a **single, static pre-allocated buffer** on both the training side and the vLLM worker side. 

By reusing the exact same static buffer tensor every step:
- The buffer will always have the exact same Virtual Address.
- oneCCL will register the IPC handle exactly once on Step 0.
- oneCCL will achieve a 100% cache hit rate for the rest of the run.
- The external memory growth caused by the weight sync broadcast will drop to exactly **zero**.

## Implementation Plan

### 1. Training Recipe (`recipes/dev/grpo_full_finetune_distributed_xpu.py`)

**Modifications in `_bg_xccl_broadcast` (and surrounding class):**
- Introduce a lazy-initialized static buffer: `self._xccl_bcast_buf`.
- During `_bg_xccl_broadcast`, instead of allocating `gpu_temp = torch.empty(...)`, check if `self._xccl_bcast_buf` exists and is large enough. If not, allocate it using `_BATCH_MAX_NUMEL`.
- For each chunk, use a view of the static buffer: `view = self._xccl_bcast_buf[:n]`.
- Copy the CPU batch into the view: `view.copy_(cpu_flat)`.
- Broadcast the view: `bg_pg.broadcast(view, root=0).wait()`.

### 2. vLLM Worker (`torchtune/dev/vllm_weight_sync_worker.py`)

**Modifications in `receive_weights_xccl_streaming`:**
- The manifest includes `batch_max_numel`.
- Introduce a lazy-initialized static buffer: `self._xccl_recv_buf`.
- Check if `self._xccl_recv_buf` exists and is equal to `batch_max_numel`. If not, allocate it: `self._xccl_recv_buf = torch.empty(batch_max_numel, device=self._xccl_device, dtype=torch.bfloat16)`.
- Instead of allocating `recv_buf` inside the `while` loop, use a view of the static buffer: `recv_buf = self._xccl_recv_buf[:batch_numel]`.
- Receive the broadcast into the view: `broadcast(recv_buf, root=0).wait()`.
- Copy the data out of the view to recreate the individual parameters (the current logic already handles this correctly by slicing `recv_buf[offset:offset+n]`).

### 3. Cleanup

- Ensure that the static buffers are properly deleted in `close_xccl_communicator` or the class destructor to prevent memory leaks if the communicator is re-initialized.

## Expected Outcome
- The external memory growth of ~30 MiB/step will be eliminated.
- The 32B model will be able to train indefinitely without exhausting the L0 free memory or requiring a manual checkpoint-restart every 80 steps.
- Performance will remain identical or slightly improve due to the removal of dynamic allocations.
