# XPU Sleep/Wake for Colocated vLLM + FSDP Training

Enables time-multiplexed colocation of vLLM inference and FSDP training on
the same Intel XPU tiles. vLLM's `sleep()`/`wake_up()` is CUDA-only upstream;
this implementation provides equivalent semantics on XPU via managed tensors.

## Problem

vLLM's `CuMemAllocator` uses CUDA driver virtual memory APIs
(`cuMemCreate`/`cuMemMap`/`cuMemUnmap`) that don't exist on XPU.
Without sleep/wake, large models (32B+) can't colocate on the same tiles
because model weights + KV cache + FSDP training state exceed 64 GiB.

Additional constraint: `torch.xpu.empty_cache()` leaks UR handles in Level
Zero when combined with FSDP `storage.resize_()` -- must never be called.

## Approach: Managed Tensor

Operates at the Python tensor level instead of the allocator level:

- **sleep()**: Back up GPU tensors to CPU via `.cpu()`, then release GPU
  storage via `untyped_storage().resize_(0)`. The caching allocator returns
  blocks to its free list, making them available for FSDP.
- **wake_up(tags)**: `untyped_storage().resize_(original_size)`, then
  `copy_()` from CPU backup. Free CPU copies.
- **No `empty_cache()`** anywhere in the cycle.

### Why `untyped_storage().resize_(0)` and not `tensor.data = torch.empty(0)`

The latter does NOT release storage -- the caching allocator keeps the old
allocation. `resize_(0)` explicitly returns blocks to the free list.

## Module

`torchtune/dev/xpu_sleep/` (3 files):

| File | Purpose |
|------|---------|
| `allocator.py` | `XPUMemAllocator` singleton: register, sleep, wake_up |
| `patch.py` | `patch_vllm_for_xpu_sleep()`: monkey-patches vLLM |
| `__init__.py` | Public API |

### Patches applied by `patch_vllm_for_xpu_sleep()`

1. `XPUPlatform.is_sleep_mode_available()` -> `True`
2. `XPUWorker.sleep()` / `wake_up()` -> delegate to `XPUMemAllocator`
3. `XPUWorker._maybe_get_memory_pool_context()` -> use `XPUMemAllocator`
4. `XPUWorker.initialize_from_config()` -> register KV caches after creation
5. `Worker.load_model()` -> register weights after loading

## GRPO Recipe Integration

Mode: `vllm_mode="colocate_sleep"` in `grpo_full_finetune_distributed_xpu.py`
Config: `recipes/configs/dev/qwen3B_grpo_colocate_sleep_xpu.yaml`

### Cycle per training step

```
1. wake_up(tags=["weights"])         -- restore vLLM weight storage from CPU
2. _sync_colocated_weights()         -- copy updated FSDP params to vLLM
3. wake_up(tags=["kv_cache"])        -- reallocate KV cache (zeroed)
4. vLLM generation (all ranks)       -- each rank generates its share
5. sleep(level=1)                    -- offload weights to CPU, release all GPU storage
6. FSDP training (fwd, bwd, optim)   -- full GPU memory available
7. goto 1
```

### Key implementation details

- `_init_vllm_early()` calls `patch_vllm_for_xpu_sleep()` before creating LLM
  and passes `enable_sleep_mode=True`
- Weight sync happens during wake_up phase (not post-optimizer), so vLLM gets
  the latest FSDP weights before generating
- First iteration skips wake_up (vLLM starts awake with checkpoint weights)

## Test Results

### Unit test: sleep/wake cycles (Qwen2.5-3B)

**Script:** `recipes/dev/_test_xpu_sleep.py`

| Config | Sleep | Freed | Wake weights | Wake KV | Correctness |
|--------|-------|-------|-------------|---------|-------------|
| TP=1, 5 cycles | 0.7-0.85s | 30.5 GiB | 0.16s | <1ms | PASS (greedy match) |
| TP=4, 3 cycles | 0.19-0.42s | 30.3-30.5 GiB | 0.16s | <1ms | PASS |

- 20 GiB training allocation succeeds during sleep phase
- No memory leak over 5 cycles (alloc/reserved stabilize at cycle 3)
- `reserved` stays high (caching allocator) but `alloc` drops correctly

### GRPO benchmark: Qwen2.5-3B, 10 tiles, 5 steps

**Date:** 2026-03-30 | **Node:** x4418c6s1b0n0

| Mode | Tiles | Steady-state | Gen speed | Sleep/Wake |
|------|-------|-------------|-----------|------------|
| `colocate` (non-sleep) | 10 training | **~7.3 s/step** | ~170 tok/s/rank | N/A |
| `colocate_sleep` | 10 training | **~8.2 s/step** | ~170 tok/s/rank | 0.9s wake+sync, 0.9s sleep |
| `server` (HTTP) | 2 vLLM + 10 train | **25.6 s/step** | ~180 tok/s (TP=2) | N/A |

Sleep mode adds ~1s overhead per step (0.9s sleep + 0.9s wake/sync) vs
non-sleep colocate. For 3B, the model is small enough to stay on tile, so
non-sleep is faster. For 32B, sleep is essential -- the model is too large
to fit alongside FSDP training state.

### Memory profile: colocate_sleep, 10 tiles

| Phase | GPU alloc (per tile) |
|-------|---------------------|
| After vLLM init (gpu_mem=0.5) | ~36-46 GiB (model + KV cache) |
| After sleep | ~1.2 GiB (residual) |
| During FSDP training | ~20-24 GiB (shards + optimizer + activations) |
| After wake_up + sync | ~36-46 GiB |

## Constraints

1. Must pre-register `torchtune` in `sys.modules` to avoid torchao XCCL USM bug
2. Must pass `enable_sleep_mode=True` to vLLM `LLM()` constructor
3. Never call `torch.xpu.empty_cache()` -- leaks UR handles with FSDP
4. `gpu_memory_utilization` can be higher with sleep (0.5 vs 0.10) since GPU
   is fully freed during training
5. CPU memory: ~6 GiB/tile for weight backup (3B model). 12 tiles = 72 GiB.
   Aurora nodes have 1 TiB RAM -- easily fits.

## Usage

```bash
# Launch with colocate_sleep config
bash recipes/dev/run_grpo_colocate_xpu.sh 10 /path/to/model 10 \
    recipes/configs/dev/qwen3B_grpo_colocate_sleep_xpu.yaml

# Or override mode on existing config
bash recipes/dev/run_grpo_colocate_xpu.sh 10 /path/to/model 10 \
    recipes/configs/dev/qwen3B_grpo_colocate_xpu.yaml \
    --config.vllm_mode=colocate_sleep --config.vllm_gpu_memory_utilization=0.5
```
