# XPU UR_RESULT_ERROR_OUT_OF_RESOURCES: `empty_cache()` + FSDP `storage.resize_()` Leak

**Status**: ROOT CAUSE IDENTIFIED, ALL ALLOCATOR MITIGATIONS EXHAUSTED, ACCEPT CPU OFFLOAD FOR 72B (2026-04-04)

**Revalidated 2026-04-23 on frameworks/2025.3.1 (torch 2.10.0a0+git449b176, Level Zero 1.24.0, I915_25.2.29): bug still active, identical signature and crash counts.** FSDP2+RL+`empty_cache()` crashes at iter ~70-75 with `UR_RESULT_ERROR_OUT_OF_RESOURCES`; FSDP1 crashes at iter ~145-150; workaround (no `empty_cache()` in FSDP loops) stable through 250 iterations. Frame-version upgrade does NOT fix this. Logs: `experiments/empty_cache_revalidate/`.

## Summary

When using PyTorch FSDP (both FSDP1 and FSDP2) on Intel Data Center GPU Max 1550 (XPU), calling `torch.xpu.empty_cache()` between FSDP forward passes causes `UR_RESULT_ERROR_OUT_OF_RESOURCES` after a deterministic number of iterations. The root cause is the interaction between `empty_cache()` and FSDP's `storage.resize_()` cycle: each `zeMemAllocDevice`/`zeMemFree` cycle through Level Zero leaks a UR handle.

**The workaround is: never call `torch.xpu.empty_cache()` in FSDP training loops.** The caching allocator reuses blocks from its free pool without touching Level Zero, preventing the leak entirely. This has been verified stable at 200+ iterations for small models (~3B params).

**However, this workaround is insufficient for large models (72B+).** Without `empty_cache()`, the XPU caching allocator accumulates 20+ GiB of fragmented reserved-but-unused blocks during FSDP AllGather/reshard cycles. On 48 GiB tiles, this fragmentation causes OOM even though actual allocated memory is only ~25 GiB. Large models *require* periodic `empty_cache()` calls to defragment, creating a direct conflict with this bug. See "Large Model Impact" section below.

The leak does **not** occur with:
- FSDP + RL pattern, **without** `empty_cache()` calls (200+ iterations stable)
- No FSDP + any pattern including `empty_cache()` (200+ iterations stable)
- FSDP + simple forward/backward with `empty_cache()` (500+ iterations stable)
- Raw `dist.all_gather()` + `storage.resize_()` + multi-stream events (20,000+ ops stable)

The crash requires **all three**: FSDP (with `storage.resize_` cycles) + multiple forward passes per step (RL pattern) + `empty_cache()` calls between forwards.

This affects reinforcement learning workloads (GRPO, PPO) where the pattern is:
1. Multiple no_grad forward passes (generation, logprob computation)
2. One grad-enabled forward + backward (policy gradient update)

These workloads naturally call `empty_cache()` between forward passes to manage memory for large generation buffers.

## Environment

- **Hardware**: Intel Data Center GPU Max 1550 (Ponte Vecchio), 64 GiB HBM/tile
- **System**: ALCF Aurora HPC
- **OS**: SLES 15 SP4, kernel 5.14.21-150400.24.55-default
- **i915 Driver**: I915_25.2.29_PSB_250224.35
- **Level Zero**: 1.24.0.0-i1146
- **PyTorch**: 2.10.0a0+git449b176 (Aurora frameworks 2025.3.1)

## Reproduction

Self-contained script: `recipes/dev/repro_xpu_resource_leak.py`

```bash
# Triggers the bug — FSDP2, RL pattern (~70 iterations):
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  repro_xpu_resource_leak.py --fsdp --layers 12 --hidden 1024 --heads 8

# Triggers the bug — FSDP1, RL pattern (~145 iterations, ~2x slower leak):
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  repro_xpu_resource_leak.py --fsdp1 --layers 12 --hidden 1024 --heads 8

# WORKAROUND — skip empty_cache() in chunked forwards (200+ iterations stable):
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  repro_xpu_resource_leak.py --fsdp --layers 12 --hidden 1024 --heads 8 \
  --no-empty-cache-in-chunks

# Does NOT trigger (500+ iterations stable) — FSDP2 + simple fwd/bwd:
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  repro_xpu_resource_leak.py --fsdp --simple --layers 12 --hidden 1024 --heads 8

# Does NOT trigger (500+ iterations stable) — no FSDP:
ZE_AFFINITY_MASK=0 python3 repro_xpu_resource_leak.py --layers 12 --hidden 1024 --heads 8
```

### What the default (RL) pattern does per iteration

```python
# Step 1: no_grad forward (simulate generation)
with torch.no_grad():
    gen_logits = policy_model(input_ids)  # FSDP allgather + forward

# Step 2: no_grad forward (compute logprobs)
with torch.no_grad():
    chunk_logits = policy_model(input_ids[chunk])  # FSDP allgather + forward

# Step 3: grad-enabled forward + backward (policy update)
logits = policy_model(input_ids)  # FSDP allgather + forward
loss.backward()                    # FSDP reduce-scatter + backward
optimizer.step()
```

Each iteration does ~3 FSDP allgathers. The `--simple` mode does only 1 allgather + 1 reduce-scatter per iteration and never crashes.

### Crash iteration counts

**FSDP2 (`fully_shard` composable API):**

| Model size | FSDP units | Seqs/iter | Crash iteration | Total `empty_cache()` calls | Error type |
|-----------|------------|-----------|-----------------|----------------------------|------------|
| 12L/1024h (0.4 GiB) | 13 | 4 | ~70 | ~210 | UR_RESULT_ERROR_OUT_OF_RESOURCES |
| 36L/2048h (3.9 GiB) | 37 | 4 | ~13* | ~39 | UR_RESULT_ERROR_OUT_OF_RESOURCES |
| 36L/2048h (3.9 GiB) | 37 | 16 | ~5* | ~15 | UR_RESULT_ERROR_OUT_OF_RESOURCES |
| **80L Qwen2.5-72B** | **81** | **4** | **step 2** | **~4** | **GPU segfault (NotPresent/PML5)** |

**FSDP1 (`FullyShardedDataParallel` wrapper API):**

| Model size | FSDP units | Seqs/iter | Crash iteration | Total `empty_cache()` calls | Error type |
|-----------|------------|-----------|-----------------|----------------------------|------------|
| 12L/1024h (0.4 GiB) | 13 | 4 | ~145 | ~435 | UR_RESULT_ERROR_OUT_OF_RESOURCES |

FSDP1 leaks at roughly **half the rate** of FSDP2, but still crashes. Without FSDP (DDP or single-device), the same RL pattern runs 500+ iterations with no issues.

**Key observation**: The leak rate scales with the number of FSDP units. With 81 FSDP units (72B model), only **~4 cumulative `empty_cache()` calls** (2 steps × 2 calls/step) are needed to trigger a crash. Each `empty_cache()` call cycles through all 81 FSDP units' storage, leaking proportionally more handles per call.

*From full GRPO workload (includes additional overhead from vLLM, chunked forwards, etc.)

## Error Messages

The crash manifests in three forms, depending on model size and leak stage:

**Form 1** — UR handle exhaustion (small models, gradual leak):
```
RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
```

**Form 2** — Page table corruption at PDE level (medium models):
```
Segmentation fault from GPU at 0xff000004XXXXXXXX, ctx_id: 1 (CCS)
  type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
Abort was called at 288 line in file:
  .../intel-compute-runtime-.../shared/source/os_interface/linux/drm_neo.cpp
```

**Form 3** — Page table corruption at PML5 level (large models / 72B, fast leak):
```
Segmentation fault from GPU at 0xff06001ba9400000, ctx_id: 1 (CCS)
  type: 0 (NotPresent), level: 4 (PML5), access: 0 (Read), banned: 1, aborting.
```
This form occurs with 72B models where the leak is fast enough (81 FSDP units per `empty_cache()` call) that Level Zero's page tables are corrupted before the UR handle counter overflows. The GPU tries to read from a freed virtual address, hitting a PML5 (5-level page table) fault.

## Workaround

**Remove all `torch.xpu.empty_cache()` calls from the training loop.** Without `empty_cache()`, the caching allocator reuses memory blocks without cycling through Level Zero, preventing the UR handle leak.

### Tradeoff

Higher peak memory usage — cached but unused blocks aren't returned to the device. The caching allocator holds freed blocks in its free pool for reuse instead of releasing them to Level Zero. This means:
- Steady-state memory after step 1 is stable (allocator reuses same blocks)
- Peak (high-water mark) memory is higher since freed blocks remain cached
- May require reducing batch sizes, generation lengths, or `vllm_gpu_memory_utilization` on memory-constrained configs
- `gc.collect()` still works to release Python-side references, allowing the caching allocator to reuse blocks sooner

For Qwen 2.5-3B on 64 GiB tiles, peak reserved memory is ~24-29 GiB with the workaround — well within budget.

### Workaround Fails for Large Models (72B+)

For Qwen2.5-72B on 48 GiB tiles (36-way FSDP, 3 training nodes), the workaround creates an **unsolvable conflict**:

- **Without `empty_cache()`**: XPU allocator fragmentation accumulates 20+ GiB of reserved-but-unused blocks. Peak allocated = 24.90 GiB (fits in 48 GiB), but peak reserved = 45.66 GiB. Step 1 OOMs because optimizer states (+3.77 GiB) push reserved past 48 GiB.
- **With `empty_cache()`**: Defragmentation works — step 1 completes with peak reserved = 37.68 GiB (10 GiB headroom). But the UR handle leak causes GPU segfault at step 2 after ~4 cumulative `empty_cache()` calls.

The fragmentation breakdown (72B, 36-way FSDP, step 0):

| Phase | allocated (GiB) | reserved (GiB) | fragmentation gap |
|-------|----------------|----------------|-------------------|
| pre_forward | 7.60 | 8.34 | 0.74 |
| post_forward | 9.19 | 29.03 | 19.84 |
| post_defrag | 9.19 | 9.88 | 0.69 |
| post_backward | 11.37 | 37.37 | 26.00 |
| post_optimizer | 15.14 | 37.61 | 22.47 |

The 20+ GiB gap is caused by FSDP AllGather/reshard cycles: each of the 80 layers AllGathers its shard (~1.8 GiB), uses it for forward, then reshards — but the allocator retains the freed 1.8 GiB block in its cache rather than reusing it for the next layer's AllGather (due to size/alignment mismatch or fragmentation).

**This makes the UR handle leak a blocking issue for large-model FSDP training on XPU, not just a nuisance.**

### Additional Constraints Discovered (72B Testing)

1. **`empty_cache()` during active FSDP forward is immediately fatal.** Calling `empty_cache()` between chunked forward passes (where FSDP has in-flight AllGather buffers) causes instant GPU segfault — not a gradual leak. FSDP's internal UR handles reference the freed blocks.

2. **`empty_cache()` between model calls is safe per-call but accumulates.** Calling between policy→ref forward or between steps (no active FSDP ops) works correctly but contributes to the cumulative UR handle leak.

3. **Leak rate scales with FSDP units, not iterations.** The relevant metric is total `empty_cache()` calls × FSDP units, not iterations. With 81 FSDP units, each call leaks ~81× more handles than with 13 units.

### Verification Results

**Reproduction script** (`recipes/dev/repro_xpu_resource_leak.py`):

| Condition | Iterations | Memory | Result |
|-----------|-----------|--------|--------|
| FSDP2 + RL + `empty_cache()` in chunks | ~70 | 1.3/5.4 GiB | **CRASH** |
| FSDP2 + RL + `--no-empty-cache-in-chunks` | 200 | 1.3/5.4 GiB (constant) | **STABLE** |
| FSDP1 + RL + `empty_cache()` in chunks | ~145 | — | **CRASH** |
| No FSDP + RL + `empty_cache()` | 200 | — | **STABLE** |

**Full GRPO recipe** (`recipes/dev/grpo_full_finetune_distributed_xpu.py`):

| Config | Tiles | Steps | Step Time | Result |
|--------|-------|-------|-----------|--------|
| Qwen 2.5-3B, grpo_samples=4 | 2 | 20 | ~5.3 s | **STABLE** (no `empty_cache()`) |
| Qwen 2.5-32B, 12 tiles | 12 | 20+ | ~25.6 s | **STABLE** (no `empty_cache()`, fits in memory) |
| Qwen 2.5-72B, 36 tiles, no `empty_cache()` | 36 | 0 | — | **OOM** at step 1 (fragmentation) |
| Qwen 2.5-72B, 36 tiles, 2 `empty_cache()`/step | 36 | 1 | 60.7 s | Step 1 OK, **CRASH** at step 2 |
| Qwen 2.5-72B, 36 tiles, 1 `empty_cache()`/step | 36 | ? | — | **UNTESTED** (v6, job expired) |

Previously crashed at step 5-13 depending on configuration. After removing all `empty_cache()` calls: stable through entire training run for models that fit without defragmentation.

### How the Fix Was Applied

In `recipes/dev/grpo_full_finetune_distributed_xpu.py`:
- `device_empty_cache()` wrapper: now skips on all XPU (not just colocate mode)
- `_safe_empty_cache()`: returns after `synchronize()` on XPU, no `empty_cache()`
- `_serialized_empty_cache()`: early-returns with barrier on XPU
- All direct `torch.xpu.empty_cache()` calls in the training loop: replaced with `gc.collect()` only or removed entirely

## Ruled Out

| Attempted mitigation | Result |
|---------------------|--------|
| `torch.xpu.empty_cache()` every iteration | **CAUSES the leak** (see Root Cause) |
| `gc.collect()` every iteration | No effect on UR leak or fragmentation |
| `torch.compile` (per-layer, inductor backend) | No effect |
| `TORCH_XPU_ALLOC_CONF=expandable_segments:True/False` | No effect — wrong env var (silently ignored) |
| `TORCH_XPU_ALLOC_CONF=garbage_collection_threshold:0.6` | No effect — wrong env var (silently ignored) |
| `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.4` (v6c) | No effect — XPU allocator does not implement GC threshold |
| `PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6` (v6d) | No effect — XPU allocator does not read any config options |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` (v6b) | CCL RDMA crash at init — virtual memory pointers incompatible with RDMA |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` | No effect |
| `NEOReadDebugKeys=1 OverrideMaxNumberOfHandles=1000000` | No effect |
| `ZEX_NUMBER_OF_CCS=0:1` | No effect |
| Math-only SDPA | No effect |
| Reducing `empty_cache()` calls to 2/step | Crashes at step 2 (same as 3/step) |
| Reducing `empty_cache()` calls to 1/step | Untested (job expired) |
| `empty_cache()` between forward chunks (during FSDP) | **Immediate segfault** — corrupts active UR handles |
| `forward_batch_size >= grpo_samples` (no chunking) | Avoids mid-forward crash but doesn't fix cumulative leak |

## Root Cause: `torch.xpu.empty_cache()` + FSDP `storage.resize_()` Interaction

**The root cause is `torch.xpu.empty_cache()` combined with FSDP's `storage.resize_(0)` / `storage.resize_(size)` cycle.**

When FSDP reshards parameters after a forward pass, it calls `storage.resize_(0)` which returns the backing memory to PyTorch's caching allocator free pool. If `empty_cache()` is then called (e.g., between chunked forward passes in an RL loop), the caching allocator releases those blocks back to Level Zero. When FSDP unshards for the next forward (`storage.resize_(size)`), it must re-acquire from Level Zero — and **each acquire/release cycle leaks a UR handle** at the Level Zero runtime layer.

Without `empty_cache()`, the caching allocator reuses blocks from its free pool without touching Level Zero, so no UR handles leak.

### Definitive Evidence

| Condition | Result |
|-----------|--------|
| FSDP + RL pattern + `empty_cache()` | **CRASH at ~70 iters** |
| FSDP + RL pattern, NO `empty_cache()` | **STABLE 200+ iters** |
| No FSDP + RL + `empty_cache()` | **STABLE 200+ iters** |
| FSDP + simple fwd/bwd + `empty_cache()` | **STABLE 500+ iters** |
| Raw `dist.all_gather()` + `empty_cache()` | **STABLE 20,000+ ops** |

The crash requires all three: FSDP (with `storage.resize_` cycles) + multiple forward passes per step (RL pattern) + `empty_cache()` calls between forwards.

### Why RL Workloads Are Uniquely Affected

Standard training does one forward + one backward per step, so FSDP reshards/unshards once per step with the caching allocator efficiently reusing the same blocks. RL workloads do multiple forward passes (generation, logprob computation) with `empty_cache()` between them (to manage memory for large generation buffers), causing repeated alloc/dealloc cycles through Level Zero.

## Analysis Details

The leak is at the **Level Zero runtime layer** (UR handle lifecycle during memory alloc/dealloc), triggered by the specific combination of FSDP's storage management and `empty_cache()`.

### Isolation tests prove driver primitives are stable

We systematically tested each XPU primitive that FSDP uses, in isolation and combination:

| Test | Ops completed | Result |
|------|--------------|--------|
| Raw `dist.all_gather()` under `no_grad` | 20,000 | **STABLE** |
| `torch.Event` / `stream.record_event()` | 10,000 | **STABLE** |
| Multi-stream allgather + events | 10,000 | **STABLE** |
| `storage.resize_(0)` / `resize_(size)` cycles | 2,400 | **STABLE** |
| Allgather + storage.resize_ (FSDP memory pattern) | 2,400 | **STABLE** |
| Full FSDP-like simulation (4 fwd × 12 layers, allgather + multi-stream + resize) | 9,600 | **STABLE** |
| **Actual FSDP2** (`fully_shard`) with RL pattern | **~140** | **CRASH** |
| **Actual FSDP1** (`FullyShardedDataParallel`) with RL pattern | **~290** | **CRASH** |

Test scripts: `recipes/dev/test_raw_allgather_leak.py`, `recipes/dev/test_event_leak.py`, `recipes/dev/test_storage_resize_leak.py`

All tests run on the same node, same session, confirming the environment is identical.

### What FSDP does differently

The FSDP-like simulation replicates FSDP's observable behavior: allgather into resizable storage across multiple streams with event synchronization, per-layer unshard/reshard cycles. Yet it runs 9,600 cycles while actual FSDP crashes at ~140.

The difference must be in FSDP's internal abstractions:
- **`FSDPParam` / `DTensor` wrappers**: FSDP manages parameters through `FSDPParam` objects that create `DTensor` views, manage `_sharded_param_data` / `_unsharded_param`, and use `init_unsharded_param()` — none of which our simulation uses
- **`torch.autograd._unsafe_preserve_version_counter`** context in `wait_for_unshard()` — interacts with autograd internals even under `no_grad`
- **`foreach_all_gather_copy_out`**: FSDP's copy-out uses a custom registered op (`torch.ops.fsdp.all_gather_copy_in`) that may allocate XPU resources differently
- **`AllGatherState` deferred cleanup**: In the forward path, FSDP defers freeing the previous allgather result to overlap with the next allgather (line 404 in `_fsdp_param_group.py`). The last layer's state is cleaned up in `finalize_backward()`, which **only runs after backward**. In no_grad mode, backward never runs.

### Why FSDP2 leaks faster than FSDP1

FSDP2's composable API uses per-layer `fully_shard()` with individual `FSDPParamGroup` objects and stream management per group. FSDP1's wrapper API may batch operations differently. Both leak, but FSDP2's finer granularity creates more intermediate objects per forward pass.

### Key evidence
- Memory usage (`torch.xpu.memory_allocated()`) is stable throughout — this is **not** a standard memory leak
- The leaked resource is internal to the Level Zero runtime (UR handles, command lists, or event objects) but is triggered by FSDP's specific tensor/storage management patterns
- **No FSDP = no leak**: DDP or single-device training with the identical RL pattern runs 500+ iterations without issue

## Impact

**PARTIALLY RESOLVED.** The workaround (remove `empty_cache()`) enables stable training for models that fit in memory without defragmentation:
- Qwen 2.5-3B on 2 tiles: stable, 20+ steps
- Qwen 2.5-32B on 12 tiles: stable, 20+ steps

**BLOCKING for large models.** For 72B+ models where FSDP allocator fragmentation exceeds available headroom:
- Without `empty_cache()`: OOM from fragmentation (reserved 45.66 GiB on 48 GiB tiles, allocated only 24.90 GiB)
- With `empty_cache()`: GPU segfault after ~4 calls (2 training steps)
- **No viable path** to stable 72B FSDP training on 48 GiB XPU tiles until this is fixed

The bug also blocks any future `garbage_collection_threshold`-based mitigation, since the GC threshold internally uses the same L0 `zeMemFree` path that triggers the leak.

With the workaround applied on compatible configs, GRPO training runs stably with no step limit. Config B (grpo_samples=16, max_gen=256) on 2 XPU tiles achieves ~8.0 s/step, **27% faster than 4x A100 TRL+vLLM** (10.9 s/step).

## Requested Action

**This is now a blocking issue for large-model training on Aurora.** The workaround (skip `empty_cache()`) is insufficient for 72B+ models where allocator fragmentation requires defragmentation to avoid OOM.

Since the leak is in the FSDP ↔ XPU interaction (not raw driver primitives), the fix likely involves:

1. **[CRITICAL] Fix the UR handle leak in Level Zero's memory alloc/dealloc path** — each `zeMemAllocDevice` / `zeMemFree` cycle (triggered by PyTorch's `empty_cache()` + FSDP's `storage.resize_`) leaks a UR handle. The handles should be properly released when memory is freed. The leak scales with the number of FSDP units (~81 for 72B), making it crash after just 4 `empty_cache()` calls on large models.
2. **[ALTERNATIVE] Add a defragmentation API that doesn't cycle through L0** — a "compact" or "coalesce" operation in the caching allocator that reorganizes its free pool without calling `zeMemFree`/`zeMemAllocDevice`. This would allow defragmentation without triggering the UR handle leak.
3. **[ALTERNATIVE] Improve XPU caching allocator block reuse** — the 20+ GiB fragmentation gap suggests the allocator is not efficiently reusing freed blocks from FSDP AllGather/reshard cycles. Better size-class matching or block coalescing within the allocator could reduce fragmentation enough to avoid needing `empty_cache()` altogether.
4. **Short-term workaround (available now, small models only)**: Remove `torch.xpu.empty_cache()` calls from FSDP training loops. Works for models ≤32B where fragmentation doesn't exceed tile memory.

## Investigation Methodology & Test Scripts

The investigation proceeded bottom-up: test each XPU primitive in isolation, then combinations, then actual FSDP, to narrow the root cause.

### Phase 1: Isolate driver primitives

| Script | What it tests | Result |
|--------|--------------|--------|
| `recipes/dev/test_raw_allgather_leak.py` | Raw `dist.all_gather()` under `no_grad`, with and without FSDP-like alloc/free pattern | 20,000 ops **STABLE** |
| `recipes/dev/test_event_leak.py` | `torch.Event`, `stream.record_event()`, multi-stream sync, FSDP-like allgather+events | 10,000 ops **STABLE** |
| `recipes/dev/test_storage_resize_leak.py` | `storage.resize_(0)`/`resize_(size)` cycles, allgather+resize, full FSDP simulation (4 fwd × 12 layers) | 9,600 cycles **STABLE** |

**Conclusion**: All driver primitives (allgather, events, streams, storage resize) are individually stable at 10,000+ ops. The leak is not in the driver layer.

### Phase 2: Bisect FSDP behavior

| Script | What it tests | Result |
|--------|--------------|--------|
| `recipes/dev/test_fsdp_unshard_leak.py` | Actual FSDP `fully_shard()` with bisection: unshard/reshard only, no_grad forward, grad forward, RL pattern, version_counter isolation | All tests 200-500 iters **STABLE** (no `empty_cache()`) |
| `recipes/dev/test_fsdp_memory_pressure.py` | Full RL pattern with two FSDP models, logprob ops, backward. Tests memory pressure vs tensor ops as trigger | 200 iters **STABLE** (no `empty_cache()`) |
| `recipes/dev/test_toplevel_fsdp_rl.py` | Full RL with top-level-only `fully_shard()` | 200 iters **STABLE** (no `empty_cache()`) |

**Conclusion**: FSDP with the RL pattern (multiple no_grad forwards + grad backward) is stable *without* `empty_cache()`. The leak is not in FSDP's unshard/reshard logic itself.

### Phase 3: Identify the trigger

| Script | What it tests | Result |
|--------|--------------|--------|
| `recipes/dev/test_repro_direct.py` | Exact repro model + RL loop, WITHOUT `empty_cache()` | 200 iters **STABLE** |
| `recipes/dev/test_empty_cache_leak.py` | Same RL loop WITH `torch.xpu.empty_cache()` between forwards | **CRASH at ~70 iters** |
| `recipes/dev/test_empty_cache_no_fsdp.py` | Same RL loop + `empty_cache()` but WITHOUT FSDP | 200 iters **STABLE** |

**Conclusion**: `empty_cache()` is the trigger, but only in combination with FSDP. Without FSDP, `empty_cache()` is safe. Without `empty_cache()`, FSDP is safe. The leak is in the interaction.

### Phase 4: Verify fix in production (small models)

The workaround was applied to `recipes/dev/grpo_full_finetune_distributed_xpu.py` by making all `empty_cache()` calls no-ops on XPU:
- `device_empty_cache()`: skips on XPU
- `_safe_empty_cache()`: synchronize only on XPU
- `_serialized_empty_cache()`: early return on XPU
- Direct `torch.xpu.empty_cache()` calls: removed or replaced with `gc.collect()`

Full GRPO training (Qwen 2.5-3B, Config A, 2 tiles) completed 20 steps at ~5.3 s/step with no crash.

### Phase 5: Large model testing reveals workaround insufficiency (72B)

Testing Qwen2.5-72B-Instruct on 4 nodes (3 training × 12 tiles = 36-way FSDP2, 1 vLLM node), 48 GiB/tile:

**Step 1: Confirmed fragmentation is the root cause of OOM (not FSDP wrapping)**
- FSDP wrapping: correct (81 units, per-layer, `reshard_after_forward=True`)
- Activation checkpointing: correct (all 80 layers verified)
- Peak allocated: 24.90 GiB (fits in 48 GiB)
- Peak reserved: 45.66 GiB (barely fits step 0, OOMs step 1 with optimizer states)
- Fragmentation gap: 20.76 GiB

**Step 2: Tested `empty_cache()` defragmentation strategies**

| Strategy | `empty_cache()` calls/step | Step 0 | Step 1 | Step 2 | Failure mode |
|----------|--------------------------|--------|--------|--------|-------------|
| No defrag (workaround) | 0 | OK | OOM | — | Fragmentation exceeds 48 GiB with optimizer states |
| Inter-chunk defrag (v3) | N/A | Segfault | — | — | `empty_cache()` during active FSDP forward destroys in-flight UR handles |
| Full defrag (v4): policy→ref + pre-backward + between-step | 3 | OK (60.7s) | OK (37.68 GiB peak) | Segfault | UR handle leak after ~4 calls |
| Reduced defrag (v5): pre-backward + between-step | 2 | OK | OK | Segfault | UR handle leak after ~4 calls |
| Minimal defrag (v6): between-step only | 1 | ? | ? | ? | Job expired before completion |
| expandable_segments (v6b): zero empty_cache | 0 | CRASH at init | — | — | oneCCL RDMA rejects virtual memory USM pointers |

**Key findings:**
1. `empty_cache()` **works for defragmentation** — freed 19-22 GiB per call, reduced peak reserved from 45.66 → 37.68 GiB
2. Step 1 **completes** with `empty_cache()` — 10 GiB headroom on 48 GiB tiles
3. GPU segfault occurs reproducibly at step 2 after **~4 cumulative calls** regardless of placement (pre-backward, between-step, or both)
4. `empty_cache()` during active FSDP forward (between chunked model calls) causes **immediate** segfault, not gradual leak
5. Setting `forward_batch_size >= grpo_samples` (no chunking) avoids the mid-forward crash
6. All previous tests used `TORCH_XPU_ALLOC_CONF` which is **NOT recognized by PyTorch** — `expandable_segments` and `garbage_collection_threshold` were silently disabled in every test. Must use `PYTORCH_ALLOC_CONF` instead (see "PYTORCH_ALLOC_CONF Discovery" section).
7. `PYTORCH_ALLOC_CONF=expandable_segments:True` is **fundamentally incompatible with oneCCL RDMA** on CXI fabric — crashes during first AllGather (see "expandable_segments vs CCL RDMA" section).

**Conclusion:** The UR handle leak makes `empty_cache()` unusable beyond ~4 calls for 81-unit FSDP models, while the allocator splintering makes `empty_cache()` mandatory for 72B to fit in 48 GiB. All allocator config options (`garbage_collection_threshold`, `max_split_size_mb`, `roundup_power2_divisions`) are **not implemented** in the XPU allocator (v2.8), and `expandable_segments` is incompatible with oneCCL RDMA. This is a dead end — a driver-level fix (UR handle leak) or XPU allocator upgrade (GC support) is required. **Accepted resolution: use `fsdp_cpu_offload: True` for 72B** (6.5s/step overhead, ~8% of step time).

### PYTORCH_ALLOC_CONF Discovery (2026-04-04)

All previous tests set `TORCH_XPU_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`. **This env var is not recognized by PyTorch.** The allocator only reads:
- `PYTORCH_ALLOC_CONF` (primary, all backends including XPU)
- `PYTORCH_CUDA_ALLOC_CONF` (backward compat, CUDA/HIP only)

Confirmed by grepping `c10/core/AllocatorConfig.h` — no reference to `TORCH_XPU_ALLOC_CONF` anywhere in PyTorch source. The `ExpandableSegment` class *does* exist in `libc10_xpu.so` (confirmed via `nm -C`), meaning the feature is implemented for XPU but was **never activated** in any of our tests.

This means findings 1-5 above were all tested **without** `expandable_segments` or `garbage_collection_threshold` actually enabled, despite the env var being set. The fragmentation gap of 20+ GiB was observed with the default allocator behavior.

### expandable_segments vs CCL RDMA (2026-04-04)

`expandable_segments:True` uses Level Zero virtual memory APIs (`zeVirtualMemReserve` / `zePhysicalMemCreate` / `zeVirtualMemMap`) to stitch non-contiguous physical memory blocks together under a single virtual address range. This avoids the contiguous-block allocation failures that cause the 20+ GiB fragmentation gap.

However, **oneCCL over CXI fabric (Slingshot 11) relies on RDMA** (Remote Direct Memory Access). RDMA network interface cards require memory regions to be:
1. **Physically contiguous** — the NIC does DMA directly to physical pages
2. **Pinned (registered)** — pages must be locked in physical memory and registered with the NIC

When FSDP passes a virtually-stitched, non-standard USM pointer to oneCCL for AllGather, the RDMA registration fails because the underlying physical memory is non-contiguous. The NIC cannot perform DMA across discontiguous physical pages mapped through virtual memory, causing an immediate RECV failure and collapsing all communicators.

**Error signature** (all 36 ranks, simultaneously):
```
CCL_ERROR: entry: RECV failed. atl_status: FAILURE
dt bfloat16, cnt 34603008, buf (src: 0x55cc8f494a50, size 69206016, off 0, type: 1, ptr: 0x14366ac00000)
terminate called after throwing an instance of 'ccl::v1::exception'
```

The `type: 1` USM pointer type confirms these are device-allocated pointers (not shared or host). The crash occurs during the **first** AllGather — the very first FSDP operation after model sharding — meaning every CCL collective is affected, not just specific tensor sizes.

**Same root cause also affects vLLM**: When vLLM processes have `expandable_segments:True` set, their internal CCL (used for tensor parallelism) crashes with `oneCCL: invalid usm pointer type: unknown`.

**Implication**: `expandable_segments` cannot be used in **any** process that communicates via oneCCL on CXI fabric, whether training (FSDP AllGather/ReduceScatter) or inference (vLLM TP). This eliminates the most promising allocator-level mitigation for the fragmentation problem.

**Remaining option**: `garbage_collection_threshold` alone (without `expandable_segments`) does not change pointer types and should be CCL-safe. It forces the allocator to reclaim fragmented blocks before allocating new ones. However, this has not yet been tested with the correct env var (`PYTORCH_ALLOC_CONF`).

### XPU Allocator Does Not Read PYTORCH_ALLOC_CONF (2026-04-04)

**Critical discovery**: The XPU caching allocator on Aurora (PyTorch 2.8.0a0, `libc10_xpu.so`) is a **simplified, standalone implementation** that does NOT read `AcceleratorAllocatorConfig`. Unlike the CUDA allocator, which queries `garbage_collection_threshold`, `max_split_size_mb`, `roundup_power2_divisions`, and `expandable_segments` from the parsed config, the XPU allocator hardcodes all behavior:

| Feature | CUDA Allocator | XPU Allocator (v2.8) |
|---------|---------------|---------------------|
| `garbage_collection_threshold` | Proactive GC at threshold | **Not implemented** — no GC code path exists |
| `max_split_size_mb` | Configurable split limit | **Hardcoded** — splits if remainder > 1 MiB (large pool) or ≥ 512 B (small pool) |
| `roundup_power2_divisions` | Configurable rounding | **Hardcoded** — rounds to 512-byte multiples |
| `expandable_segments` | Implemented | **Not in v2.8** (added in main/v2.10+, but CCL-incompatible) |
| Block search | Best-fit with size classes | `lower_bound` on `(queue, size, ptr)` — first-fit by size |
| OOM recovery | Release specific blocks, GC | **Release ALL cached blocks** (`release_cached_blocks()`) |

**Confirmed via**:
- `nm -CD libc10_xpu.so` — only 9 symbols exported: `emptyCache`, `raw_alloc`, `raw_delete`, `getDeviceStats`, `init`, `recordStream`, `get`, `resetPeakStats`, `resetAccumulatedStats`. No `garbage_collect`, `release_cached_blocks`, or config-related symbols.
- XPU allocator header (`c10/xpu/XPUCachingAllocator.h`) — minimal interface, inherits from `CachingDeviceAllocator` for stats only.
- PyTorch v2.8.0 source (`c10/xpu/XPUCachingAllocator.cpp`) — `malloc()` calls `get_free_block()` → `alloc_block()` → `release_cached_blocks() && alloc_block()`. No threshold check, no config query.

**Implication**: ALL `PYTORCH_ALLOC_CONF` settings are silently ignored by the XPU allocator. Tests v6c (`garbage_collection_threshold:0.4`) and v6d (`max_split_size_mb:512,garbage_collection_threshold:0.6`) produced identical 30.25-31.37 GiB gaps because **neither setting was ever read by the allocator**. The config is parsed by `AcceleratorAllocatorConfig` but the XPU `malloc()` never queries it.

**What the XPU allocator actually does on allocation failure**:
```cpp
block_found = alloc_block(params, false) ||
    (release_cached_blocks() && alloc_block(params, true));
```
When allocation fails, it calls `release_cached_blocks()` which frees **ALL** non-split free blocks via `sycl::free()` (= `zeMemFree`). This is equivalent to `empty_cache()` — and triggers the same UR handle leak. There is no partial release, no GC threshold, no size-class-aware reclamation.

### Allocator Mitigation: Dead End (2026-04-04)

All allocator-level approaches to the 30 GiB splintering gap are exhausted:

| Approach | Why it fails |
|----------|-------------|
| `garbage_collection_threshold` | XPU allocator doesn't implement it |
| `max_split_size_mb` | XPU allocator doesn't read it |
| `roundup_power2_divisions` | XPU allocator doesn't read it |
| `expandable_segments` | (a) Not in v2.8, (b) incompatible with CCL RDMA |
| `empty_cache()` | Works for defrag but triggers UR handle leak (crashes after ~4 calls on 72B) |
| OOM-triggered release | Uses same `release_cached_blocks()` → `sycl::free()` path as `empty_cache()` — same UR leak |

**The splintering gap cannot be resolved at the allocator configuration level.** The XPU allocator is too simple (no config, no GC, no partial release), and the one operation that does release memory (`sycl::free` via `empty_cache()` or OOM retry) triggers a driver-level UR handle leak.

### Resolution: Accept CPU Offload (2026-04-04)

**Context**: The fragmentation investigation was motivated by the 72B optimization roadmap (Phase 8 in `aurora_rl_baselines.md`). With 3 training nodes (36-way FSDP), per-tile allocated memory is only ~24.9 GiB — well within 48 GiB. The goal was to eliminate CPU offload (which adds ~6.5s/step optimizer overhead) by fitting without defragmentation.

**Decision**: Accept `fsdp_cpu_offload: True` for 72B models. The 6.5s/step overhead (~8% of 84.6s step time) is not worth fighting a driver-level allocator bug. Optimization effort is better directed at:
- `forward_batch_size=4` (estimated -28s, 4× more impactful than eliminating offload)
- More training nodes (reduces both per-tile memory AND offload need)
- Scaling `grpo_samples` and `max_generated_tokens` for training quality

**This bug remains blocking for any future attempt to run 72B without CPU offload on 48 GiB tiles.** A driver fix (UR handle leak in `zeMemFree`) or an upgraded XPU allocator (with GC threshold support) would reopen this path.
