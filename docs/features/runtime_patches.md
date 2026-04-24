# Runtime Patches and Workarounds

This documents all monkeypatches applied at runtime to make GRPO training work on Aurora XPU. These exist because upstream packages (vLLM, PyTorch FSDP, Level Zero) have XPU-specific bugs or missing features. Each patch should be re-evaluated when `module load frameworks` is updated.

## Active Patches (Production)

### 1. torchtune `__init__.py` bypass (sys.modules pre-registration)

**Location**: `recipes/dev/grpo_full_finetune_distributed_xpu.py` lines 66-81

**Problem**: Running `torchtune/__init__.py` while an XCCL process group is active corrupts the Level Zero USM pointer table. The `__init__` imports `torchao` and sets env vars that interact badly with XCCL's device-context bookkeeping.

**Fix**: Pre-register `torchtune` in `sys.modules` as a plain `types.ModuleType` with `__path__` set correctly. This allows `from torchtune.xxx import ...` to work without executing `__init__.py`.

**Used by**: All training ranks (always active).

**How to verify still needed**: Remove the pre-registration block, run a 2-tile GRPO training. If XCCL collectives fail with USM pointer errors, the patch is still needed.

### 2. FSDP2 ReduceOp.AVG → SUM

**Location**: `recipes/dev/grpo_full_finetune_distributed_xpu.py` lines 115-141

**Problem**: FSDP2's `reduce_scatter` uses `ReduceOp.AVG`, which XCCL does not support.

**Fix**: Monkeypatch `_get_gradient_divide_factors()` to force `force_sum_reduction_for_comms=True` when `device_type == "xpu"`. This makes FSDP use `SUM + manual division` instead of `AVG`. Same approach used by MTIA in upstream PyTorch.

**Used by**: All FSDP2/HSDP training (always active).

**How to verify still needed**: Check if `torch.distributed.ReduceOp.AVG` works with XCCL: `dist.all_reduce(tensor, op=dist.ReduceOp.AVG)`. If it raises an error, the patch is still needed.

### 3. `device_empty_cache()` no-op on XPU

**Location**: `recipes/dev/grpo_full_finetune_distributed_xpu.py` lines 148-160

**Problem**: `torch.xpu.empty_cache()` combined with FSDP's `storage.resize_()` leaks UR (Unified Runtime) handles in Level Zero. After ~70 iterations, this causes `UR_RESULT_ERROR_OUT_OF_RESOURCES`.

**Fix**: Replace `device_empty_cache()` with a no-op for XPU. The caching allocator reuses memory blocks without touching Level Zero if `empty_cache()` is never called.

**Used by**: All XPU training (always active).

**How to verify still needed**: Run GRPO for 100+ steps with `torch.xpu.empty_cache()` re-enabled. If training crashes with UR resource errors, the patch is still needed. See `docs/bugs/intel_xpu_resource_leak_bug_report.md` for full analysis.

### 4. Transformers version check bypass

**Location**: `recipes/dev/_usercustomize_vllm/usercustomize.py` lines 18-34

**Problem**: The `frameworks` module provides `huggingface-hub>=1.7.x`, but `transformers` requires `<1.0`. vLLM imports transformers, which crashes on version check.

**Fix**: Replace `transformers.utils.versions._compare_versions()` with a no-op.

**Used by**: All vLLM server processes (via `PYTHONPATH` including `_usercustomize_vllm/`).

**How to verify still needed**: Run `python -c "import transformers"` with the frameworks module loaded. If it raises a version error about huggingface-hub, the patch is still needed.

### 5. vLLM registry subprocess fallback

**Location**: `recipes/dev/_usercustomize_vllm/usercustomize.py` lines 55-68

**Problem**: vLLM's `_run_in_subprocess()` (used by the model registry to probe model configs) segfaults on XPU.

**Fix**: Wrap the subprocess call in a try/except — on `RuntimeError`, fall back to running the function in-process.

**Used by**: All vLLM server processes.

**How to verify still needed**: Start a vLLM server without this patch. If it segfaults during model registry initialization, the patch is still needed.

### 6. vLLM XPU memory detection

**Location**: `recipes/dev/_usercustomize_vllm/usercustomize.py` lines 86-130

**Problem**: vLLM's default `determine_available_memory()` computes KV cache budget as `total(64G) * utilization - peak`, but on Aurora the Level Zero driver contexts consume ~52 GiB, making the calculation wildly wrong.

**Fix**: Replace with a version that uses `torch.xpu.mem_get_info()` to check actual free memory after model profiling, then scales by `gpu_memory_utilization`.

**Used by**: All vLLM server processes.

**How to verify still needed**: Start vLLM without the patch with `--gpu-memory-utilization 0.80`. If it OOMs or allocates incorrectly, the patch is still needed.

## Conditional Patches (Not in Default Production Path)

### 7. Gloo group/allreduce patching (colocate mode only)

**Location**: `recipes/dev/grpo_full_finetune_distributed_xpu.py` lines 486-512

**Problem**: When vLLM is colocated in the training process (not a separate server), it creates Gloo sub-groups that can't handle XPU tensors.

**Fix**: Patches `torch.distributed.new_group()` and `all_reduce()` to no-op for single-rank Gloo groups with XPU tensors.

**Used by**: Only when config has `vllm_mode: colocate` or `colocate_sleep`. **Not used** in the default production configs (which use `vllm_url` server mode).

### 8. XPU sleep/wake for colocated vLLM

**Location**: `torchtune/dev/xpu_sleep/patch.py`

**Problem**: Colocated vLLM needs to release GPU memory during training and reclaim it during generation.

**Fix**: Patches `XPUPlatform`, `XPUWorker`, and `Worker` classes to support sleep/wake callbacks via a custom `XPUMemAllocator`.

**Used by**: Only when config has `vllm_mode: colocate_sleep`. **Experimental** — 128 s/step (5x slower than server mode).

### 9. vllm_serve_xpu.py custom server (TP=1 only)

**Location**: `recipes/dev/vllm_serve_xpu.py`

**What it does**: Custom vLLM server with `/load_weights_from_path/` endpoint for file-based weight synchronization. Also applies the transformers version check patch.

**Used by**: Only when `VLLM_TILES=1` in the launch script. Default production uses `VLLM_TILES=2` (TP=2), which uses the standard `vllm.entrypoints.openai.api_server` instead. Currently **not active** in optimized configs.

## Checking If Patches Are Still Needed

When `module load frameworks` is updated to a new version:

1. Check `huggingface-hub` version: `pip show huggingface-hub` — if `<1.0`, patch 4 may be removable
2. Check XCCL `ReduceOp.AVG` support — if supported, patch 2 can be removed
3. Run 100+ step GRPO with `empty_cache()` enabled — if no UR leak, patch 3 can be removed
4. Test `torchtune` import during XCCL — if no USM corruption, patch 1 can be removed
5. Test vLLM subprocess registry on XPU — if no segfault, patch 5 can be removed
6. Test vLLM memory detection — if `mem_get_info()` reports realistic values, patch 6 can be removed

## Frameworks Version Tested

- `aurora_frameworks-2025.3.1`
- PyTorch `2.10.0a0+git449b176`
- vLLM `0.15.x` (bundled)
- Python `3.12`
