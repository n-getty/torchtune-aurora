# XPU UR_RESULT_ERROR_OUT_OF_RESOURCES at step 0 under per-tile process co-tenancy

**Status**: ROOT CAUSE CONFIRMED. Production fix = W2 (process-isolated tiles). W2 validated end-to-end on Qwen3-8B, 3/3 steps clean, exit=0 (2026-05-07).

## Summary

When two distinct OS processes attach to the same Aurora PVC tile and both submit XPU/Level-Zero work concurrently, the per-tile L0 driver context exhausts an internal event/queue/submission resource pool **at the very first backward kernel of step 0**, with **~30 GiB of HBM headroom remaining**. The error is `level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)` — bit-identical to the well-known `empty_cache()` + FSDP `storage.resize_()` leak documented in [`intel_xpu_resource_leak_bug_report.md`](intel_xpu_resource_leak_bug_report.md), but with a completely different trigger and timing.

This variant surfaced in the Ray-colocate Qwen3-8B TP=8 implementation (8 trainer ranks + 8 vLLM Ray actors sharing the same 8 tiles); see `docs/reports/colocate_ray_tp8_status_20260507.md` for the smoke matrix and full diagnosis.

## How this differs from the known UR40 bug

| Dimension | Known FSDP+empty_cache UR40 | This variant (co-tenancy) |
|-----------|------------------------------|----------------------------|
| Trigger | `torch.xpu.empty_cache()` between FSDP forwards | Two L0 driver clients on the same tile |
| When it fires | iter ~70 (FSDP2) / ~145 (FSDP1) | step 0, first BWD chunk |
| Workaround | Never call `empty_cache()` in FSDP loops | Open question — see "Mitigation candidates" |
| `device_empty_cache` involved? | Yes (the cause) | No (it's a no-op on XPU; recipe never calls real empty_cache in steady-state) |
| HBM at crash | Often near limit | ~30 GiB headroom (smoke6/7 evidence) |
| Reproducer | `recipes/dev/repro_xpu_resource_leak.py` | `experiments/colocate/run_qwen3_8b_colocate_ray.sh` (smokes 3-7, 2026-05-07) |

The two share **the same error code** (UR 40 = `UR_RESULT_ERROR_OUT_OF_RESOURCES`) and likely the **same underlying L0 resource pool**, but they are exhausted by different mechanisms:

- **Known bug**: each `empty_cache()` triggers a `zeMemFree` → `zeMemAllocDevice` cycle that leaks one UR handle per FSDP `storage.resize_()` call. Cumulative; deterministic iteration count.
- **This variant**: two processes simultaneously hold L0 driver state on the same tile. The driver allocates per-process event/queue/submission objects from a shared per-tile pool. The pool ceiling is reached during the first backward — when the trainer process's gradient accumulation kernels and the Ray actor process's idle event queues both have outstanding handles.

## Reproduction

Repo state at hash:
- Recipe: `recipes/dev/grpo_full_finetune_distributed_xpu.py:2315` (`_generate_with_ray_colocate_vllm`)
- Config: `recipes/configs/dev/experimental/qwen3_8b_grpo_colocate_ray_xpu.yaml`
  - `vllm_mode: colocate_ray`, `vllm_tensor_parallel_size: 8`
  - `vllm_gpu_memory_utilization: 0.30` (down from 0.55 — same wedge across all values)
  - `forward_batch_size: 1`, chunked-loss on
- Launcher: `experiments/colocate/run_qwen3_8b_colocate_ray.sh`

Run on a fresh debug-scaling hold:
```bash
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/run_qwen3_8b_colocate_ray.sh 3 4
```

Expected outcome: gen completes (750-880 tok/s, real reward), then `level_zero backend failed with error: 40` at the first BWD chunk of step 0 on rank 0.

## Smoke matrix (envelope sweep — same wedge across all combinations)

| Smoke | util | fwd_batch | chunked_loss | Result | Trainer pre-bwd |
|-------|------|-----------|--------------|--------|-----------------|
| 3 | 0.55 | 2 | off | UR 40 step=0 chunk[0:2] BACKWARD FAILED | alloc=9.61 / resv=16.90 GiB |
| 4 | 0.40 | 2 | off | UR 40 step=0 chunk[0:2] BACKWARD FAILED | alloc=11.51 / resv=18.08 GiB |
| 5 | 0.40 | 2 | on | True HBM OOM | PyTorch 27.58 GiB allocated; vLLM ~26 GiB; 64 GiB tile saturated |
| 6 | 0.30 | 2 | on | UR 40 SINGLE-BWD FAILED step=0 | alloc=6.29 / resv=11.89 GiB — NOT HBM OOM |
| 7 | 0.30 | 1 | on | UR 40 SINGLE-BWD FAILED step=0 | alloc=6.29 / resv=11.89 GiB — NOT HBM OOM |

Smoke6/7 evidence is conclusive: at util=0.30 vLLM holds ~22 GiB (model 1.92 + KV 18.2 + overhead), trainer post-gen sits at 6.29 GiB, leaving ~30 GiB of HBM headroom on each tile when bwd starts. The wedge is not HBM-OOM.

## Why we believe it is L0 driver, not PyTorch

1. PyTorch's allocator reports plenty of free memory (`mem_get_info` clean, `memory_reserved` < 12 GiB).
2. The error is raised by the **Level Zero backend** (`level_zero backend failed with error: 40`), not by PyTorch's allocator.
3. The bug does not occur in any of:
   - Same recipe, vLLM in `dedicated_rank` mode (no co-tenancy).
   - Same recipe, vLLM in `server` mode (separate node).
   - Standalone vLLM TP=8 generation under Ray (no trainer process).
   - Standalone trainer FSDP2 BWD on the same tiles (no Ray actor).
4. `device_empty_cache(device)` is a no-op on XPU (`torchtune/dev/rl/distributed.py:775`), so the known FSDP+empty_cache leak path is not active.
5. Co-tenancy is the only state the failing configuration has that all four passing configurations lack.

## Cross-framework prior art

Surveyed RL frameworks for the same pattern (NeMo-RL, torchrl, OpenRLHF — all use Ray-actor + vLLM sleep on a single device pool):

- **NeMo-RL** (`vllm_generation`): same Ray-actor pattern; relies on CUDA IPC + NCCL, which has no analog of L0's per-tile event-pool ceiling.
- **torchrl** (`AsyncVLLM`): defaults to **separate-device** placement; does not exercise co-tenancy.
- **OpenRLHF** (Hybrid Engine): co-locates via `VLLM_RAY_PER_WORKER_GPUS` and time-multiplexes with `--vllm.enable_sleep` + `--ds.enable_sleep`. CUDA stack — no UR40.

**Verdict**: this is an Intel L0 / Aurora-stack issue, not an RL-framework design issue. Adopting any of these patterns verbatim does not dodge the L0 wedge.

## Mitigation results (sweep complete 2026-05-07)

| ID | Lever | Result | Notes |
|----|-------|--------|-------|
| W4 | `torch.xpu.empty_cache()` between gen and bwd (`TORCHTUNE_RAY_COLOCATE_DRAIN_L0=1`) | **FAIL** — SINGLE-BWD FAILED step=0 UR40 | empty_cache does not drain enough L0 event handles to clear the per-tile pool ceiling. Rules out the theory that bwd is just one event-pool drain away from succeeding. |
| W5 | `ZEX_NUMBER_OF_CCS=0:2` | **FAIL** — SINGLE-BWD FAILED step=0 UR40 | CCS partitioning bounds compute-stream contention, not event-pool exhaustion. Different L0 resource. |
| W45 | W4 + W5 combined | Skipped (independent failures sufficient) |
| SKIPGEN | Ray actors exist but vLLM `.generate()` is bypassed (`TORCHTUNE_RAY_COLOCATE_SKIP_GENERATE=1`) | **FAIL** — SINGLE-BWD FAILED step=0 UR40 | **Critical finding**: actor *existence* (L0 driver attach + KV cache pre-allocation) is sufficient to wedge bwd. Generation activity is irrelevant. **This rules out W1** (Ray sleep/wake): vLLM 0.15.0 sleep releases tensor storage but cannot release the L0 driver context for an alive process. |
| W1 | Ray sleep/wake adapter | **RULED OUT** by SKIPGEN | Would only help if generation activity were the trigger. SKIPGEN proved otherwise. NeMo-RL's CUDA pattern does not transfer because CUDA has no analog of L0's per-tile event-pool ceiling. |
| W2 | 8+4 process-isolated tiles (vLLM TP=4 on spare tiles) | **PASS — production fix** | Trainer ranks 0..7 on tiles 0..7; Ray head + vLLM actors on tiles 8..11 (`ONEAPI_DEVICE_SELECTOR=level_zero:8,9,10,11` for the ray-start subshell only; trainer keeps full 12-tile visibility for CCL UUID discovery). Eliminates co-tenancy entirely. Validated 2026-05-07 with Qwen3-8B GRPO — 3/3 steps clean, exit=0. |

### Finding chain (2026-05-07)

```
W4 fail  → empty_cache cannot drain enough L0 events
W5 fail  → CCS partitioning is wrong resource
SKIPGEN fail → wedge is from actor existence, not generate activity
              → W1 (sleep/wake) cannot help (driver context is process-bound)
W2 pass  → only fix is to avoid co-tenancy; place trainer + vLLM on disjoint tiles
```

### Follow-on bugs uncovered by W2 (all fixed 2026-05-07)

The W2 path exposed three latent bugs that the colocated-TP=8 path had been masking by failing earlier in BWD:

1. **`_generate_with_ray_colocate_vllm` rank-0 cat dim mismatch** (`recipes/dev/grpo_full_finetune_distributed_xpu.py:2349-2369`). Each rank pads `input_ids` to its own per-rank max prompt length, so gathered tensors had different second-dim sizes. The original `torch.cat([t for t in gathered])` raised `Sizes of tensors must match...`. Fix: pad each gathered tensor to `global_ctx = max(t.shape[1] for t in gathered)` before cat; trim `[global_ctx - local_ctx]` columns out of the broadcast result on the per-rank scatter side so each rank's downstream code still sees `(bsz_local, local_ctx + max_gen)`.
2. **`qr_full` allocated with rank-local `context_length` instead of `global_ctx`** (`recipes/dev/grpo_full_finetune_distributed_xpu.py:2416-2429`). Same root cause: rank-0's `context_length` was rank-local. Fix: allocate `qr_full` with `(bsz_total, global_ctx + max_gen)` and write completions starting at column `global_ctx`.
3. **`collective_rpc("load_weights", ...)` returned `NotImplementedError`** (`torchtune/dev/rl/weight_sync.py:128-198`). vLLM v1's `RayWorkerWrapper.execute_method` does `getattr(worker, method_name)`, but `load_weights` is on `worker.model_runner.model`, not on the worker itself. Fix: pass a callable `_ray_load_weights(worker, blob)` that walks down to the model. Required two additional fixes:
   - `VLLM_ALLOW_INSECURE_SERIALIZATION=1` in the launcher env (vLLM refuses to cloudpickle a function otherwise).
   - **Tensor serialization quirk**: msgspec's encoder serializes torch tensors as `(dtype, shape, data)` tuples but the decoder only reconstructs them when the call schema is typed `torch.Tensor` (`vllm/v1/serial_utils.py:dec_hook`). In an untyped `collective_rpc` args payload the tensor arrives as a plain list and `load_weights` crashes with `'list' has no attribute 'shape'`. Fix: pickle the `[(name, tensor)]` list to a bytes blob on the trainer side and unpickle inside the worker callable.

### Reference: cross-framework patterns

NeMo-RL, torchrl, OpenRLHF all use Ray-actor + vLLM sleep on a single device pool — but on CUDA, where there is no L0 event-pool analog. None of them encountered this UR40. Adopting their patterns verbatim does not dodge the L0 wedge. Process isolation (W2) is the only mechanism that addresses the Aurora-stack constraint.

## What we need from Intel

If this turns out to be a real L0 driver ceiling on per-tile event/queue resources, the requested upstream changes are:

1. Public env knob to query and set the per-tile event-pool ceiling (today: `ZEX_NUMBER_OF_CCS` is the closest documented control, and only over CCS partitioning).
2. Diagnostic API to inspect per-process L0 resource usage (today: nothing — we can only observe `UR_RESULT_ERROR_OUT_OF_RESOURCES` post-mortem).
3. Documentation of which L0 resources are per-process vs per-tile vs per-context.

## Cross-references

- [`intel_xpu_resource_leak_bug_report.md`](intel_xpu_resource_leak_bug_report.md) — the iteration-bounded `empty_cache()` + FSDP variant. Same error code, different cause.
- [`ccl_ipc_handle_cache.md`](ccl_ipc_handle_cache.md) — CCL `banned:1` PDE from IPC handle exhaustion. Distinct from this; manifests as PDE not UR40.
- `docs/reports/colocate_ray_tp8_status_20260507.md` — the Ray-colocate smoke matrix and full diagnosis.
- `memory/project_colocate_tp8_ray_implementation.md` — running investigation history.

## Status

**OPEN.** Mitigation sweep in flight on hold 8472800 (debug-scaling, queued 2026-05-07).
