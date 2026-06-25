# XPU UR_RESULT_ERROR_OUT_OF_RESOURCES at step 0 under per-tile process co-tenancy

**Status**: ROOT CAUSE CONFIRMED. **Two viable production paths**: (a) W2 process-isolated tiles 8+4 (fastest, validated 247s/step Qwen3-8B); (b) **W17+W19 fully-colocated TP=8 kill+respawn cycle** (validated 2026-05-08, 3/3 steps clean, ~290s/step — only +17% over W2). Pick W17+W19 when HBM forces full 8-tile colocation; pick W2 when 4 spare tiles are available.

**Mechanism breakthrough (2026-05-07, W17)**: the wedge is from **LIVE concurrent co-tenancy**, not from sticky L0 state. Tearing down the vLLM Ray actors + `ray.shutdown()` AFTER generation but BEFORE the trainer's first BWD chunk **unblocks BWD** — step 0 BWD + optimizer completes cleanly on all 8 ranks, with the same trainer process and the same per-tile L0 driver state that previously wedged.

**Workaround validated (2026-05-08, W19)**: kill (W17) → BWD → opt → respawn vLLM (rank 0) **with trainer ranks parked on a CPU-only gloo barrier** while rank 0 builds the new `LLM(...)`. The naive xccl barrier (W18) deadlocked here for the *symmetric* reason: ranks 1-7 in `torch.distributed.barrier()` (xccl) became live L0 clients again, competing with the 8 vLLM TP init `all_reduce` ops. Switching to `self._ray_colocate_gen_pg` (gloo, CPU-only) lets rank 0 cold-start vLLM in 30s while peers hold zero L0 work. Three full kill+respawn cycles ran clean, ratios=1.0000, no banned:1, no UR40. Full log: `experiments/colocate/ray_colo_logs/W19_kill_respawn_20260508_001441/run.log`. Recipe hook: `recipes/dev/grpo_full_finetune_distributed_xpu.py:4150`.

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
| Reproducer | `experiments/arena_ipc/repro_xpu_resource_leak.py` | `experiments/colocate/run_qwen3_8b_colocate_ray.sh` (smokes 3-7, 2026-05-07) |

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
| W6 | NEO `NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=1 ForceInOrderImmediateCmdListExecution=1 EnableTimestampPoolAllocator=1` | **FAIL** — Rank 2 SINGLE-BWD FAILED step=0 UR40 (job 8474314, log `ray_colo_logs/W6_counterevents_20260507_210116/run.log`) | Counter-based events were the most promising spec-level structural fix (`ZE_EVENT_POOL_COUNTER_BASED_EXP` since L0 1.10) — designed to side-step the per-tile event-pool ceiling. NEO knobs are confirmed honored by the i915_25.2.29 driver (sanity check: `PrintDebugSettings=1` echoed all three back as "Non-default value of debug variable"). Gen still completes cleanly (real reward 8.71 mean, 0.80 success), then UR40 fires at first BWD chunk. Memory at crash: alloc=6.28 GiB, resv=10.63 GiB — same low-headroom signature as W4/W5/SKIPGEN. **Implication**: the wedged L0 resource is **NOT the event pool**. Signature #1 is not event-pool-bound the way ALCF Bug 106 / GSD-12152 suggested. Co-tenancy fix remains structural (W2). |
| W7 | `EnablePidFdOrSocketsForIpc=0` + `MakeEachAllocationResident=1` | **N/A** — vLLM init OOM before reaching BWD (log `ray_colo_logs/W7_ipc_resident_20260507_210857/run.log`) | `MakeEachAllocationResident=1` makes every `zeMemAllocDevice` call `zeContextMakeMemoryResident`. On TP=8 vLLM init this OOMs on every 2 MiB alloc despite 63.65 GiB free per tile — the residency pin is too aggressive for the workload. Knobs themselves are honored. Need to split: W8 isolates the IPC change. |
| W8 | `EnablePidFdOrSocketsForIpc=0` (IPC backing only) | **FAIL** — Rank 0 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W8_ipc_only_20260507_211144/run.log`) | Switching the L0 IPC handle backing from socket-based to pid+fd does not move the symptom. Combined with W6 (event pool ruled out), the wedged L0 resource is **neither the event pool nor the IPC handle table**. Most likely candidates remaining: command-list pool, command-queue pool, submission objects, or USM context allocator state. |
| W9 | `UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4096` (16× the documented default of 256) | **FAIL** — Ranks 3,4 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W9_event_pool_ceiling_20260507_211856/run.log`) | Even allowing 16× more events per pool does not move the symptom. **Conclusively rules out event-pool count** as the limiting resource (W6 ruled out the mechanism via counter-based events; W9 rules it out via raw count). Three independent attacks on the obvious candidates — events (W6/W9), IPC handles (W8) — have all failed. The wedged resource is something else entirely: command-list pool, command-queue pool, submission objects, USM context allocator state, or per-process driver bookkeeping. |
| W10 | `MakeEachEnqueueBlocking=1` (every kernel/transfer enqueue blocks until completion; zero outstanding submissions per process at any moment) | **FAIL** — Rank 3 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W10_blocking_enqueue_20260507_212503/run.log`). Knob confirmed honored via `PrintDebugSettings=1`. | **Most diagnostically valuable result.** With zero outstanding submissions per process, the wedge still fires at step-0 BWD. This **rules out race / outstanding-submission-accumulation pressure** as the mechanism. Combined with W4/W5/W6/W8/W9 (all attacks on dynamic accumulation candidates), the conclusion is now: **the wedge is from the STATIC process-context footprint of two L0 driver clients sharing a tile**, not from any in-flight resource buildup. **No application-level pacing, batching, throttling, or recipe-side fix can address this.** Only process isolation (W2) or an Intel driver fix is viable. |
| W11 | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` + `UR_L0_USE_DRIVER_INORDER_LISTS=1` (regular command lists with driver-side in-order semantics, instead of immediate command lists) | **FAIL but FAULT MODE SHIFTS** — `Segmentation fault from GPU at 0x0, ctx_id:1 (CCS) NotPresent PML4 Atomic, banned:1` followed by NEO i915 `drm_neo.cpp:288` abort, SIGABRT (exit -6), at first BWD step=0 (log `ray_colo_logs/W11_cmdlist_mode_20260507_213456/run.log`). Gen completes cleanly first. | **First probe to change the failure class.** W6/W8/W9/W10 all failed with the same UR40 = `UR_RESULT_ERROR_OUT_OF_RESOURCES`. W11 fails with a different fault — a null-pointer atomic page fault on a CCS GPU context, not a UR-layer resource exhaustion. This implicates **command-list mode / objects** in the static driver footprint that wedges the tile under co-tenancy. The "regular + driver-inorder" path exposes a different driver bug (atomic-on-NotPresent-PML4) rather than fixing the underlying co-tenancy issue, but it provides the first positive signal differentiating one driver-side resource class from the others. Production fix remains W2 (process isolation). |
| W15 | `ZEX_NUMBER_OF_CCS=0:4,1:4,…,11:4` — CCS=4 mode applied to ALL 12 root devices for both trainer and Ray actor processes. **Re-do of W5 with correct multi-tile syntax** per [intel/compute-runtime MULTI_CCS_MODES.md](https://github.com/intel/compute-runtime/blob/master/level_zero/doc/experimental_extensions/MULTI_CCS_MODES.md): W5 used `0:2`, which only set CCS mode on root device 0 (tile 0); tiles 1-7 stayed at default 1 CCS. The Intel doc explicitly warns that mismatched CCS modes between co-tenant processes "block the second application's submissions until the first finishes" — exactly the wedge symptom. | **FAIL** — Ranks 0,1,2 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W15_ccs_all_tiles_20260507_223834/run.log`). Gen completes cleanly first (success 0.81). | **Definitively rules out CCS-mode-mismatch as the wedge mechanism.** Even with CCS=4 properly applied to all 12 tiles in BOTH the trainer and Ray actor processes (matching modes, no documented stall), UR40 still fires at step-0 BWD with the bit-identical signature. The Intel-documented stall mechanism for mode mismatch was a plausible hypothesis (W5 was misconfigured), but W15 closes it: same signature even with proper config. **Combined with W11**, the implicated subsystem is the cmdlist family but NOT the CCS partition layer. |
| W16 | `UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH=4096` (16× default 256) + `UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=1` (force aggressive reuse cleanup) + `UR_L0_BATCH_SIZE=1` (minimum batch size). Targets the cmdlist family that W11 implicated. Stays on default immediate-cmdlist path (mode=1) so we test pool ceilings on the UR40-emitting path, not W11's PDE-emitting path. | **FAIL** — Rank 5 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W16_cmdlist_pool_knobs_20260507_224613/run.log`). Gen completes cleanly first (success 0.82). | **Lifting cmdlist-pool ceilings does NOT lift the wedge.** Confirms what W6 (counter events) and W9 (event-pool count) showed for events: **lifting numerical ceilings is not the answer. The wedged resource is in the EXISTENCE of the resource family, not its capacity.** Combined with W11, the implicated subsystem is the cmdlist family — but at a layer below the per-batch event ceiling, batch sizing, or cleanup. Most likely candidates remaining: the cmdlist allocator state itself, persistent submission buffers, KMD-side cmdlist bookkeeping, or fixed-size scratch reservations baked into per-process driver init. |
| W13 | `CCL_SYCL_ALLGATHERV_TMP_BUF=1` + `CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1` + `CCL_SYCL_ALLREDUCE_TMP_BUF=1` + `CCL_SYCL_BROADCAST_TMP_BUF=1` + `CCL_SYCL_REDUCE_TMP_BUF=1`. Per-collective persistent device-side temp buffers — bypass per-call L0 IPC handle exchange. Targets the layer ABOVE L0 (oneCCL temp-buf path). | **FAIL** — Ranks 0,1,3,4,6,7 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W13_oneapi_temp_buf_20260507_224918/run.log`). Gen completes cleanly first (success 0.79). | **Wedge is BELOW oneCCL — in the L0 driver layer directly**, not in the per-call IPC handle exchange that the temp-bufs were designed to bypass. Combined with W8 (IPC backing change), this is a second independent attack on the IPC-handle hypothesis from a different layer, both negative. The L0 IPC handle table family is now ruled out at two layers (UR-level via W8, oneCCL-level via W13). |
| W11a | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=2` (per-thread immediate command lists — the OTHER non-default cmdlist branch from W11) | **FAIL** — Ranks 0,3 SINGLE-BWD FAILED step=0 UR40 (log `ray_colo_logs/W11a_cmdlist_perthread_20260507_225522/run.log`). Gen completes cleanly first (success 0.79). | Mode=2 emits the SAME UR40 signature as default mode=1, NOT W11's PDE failure. So the cmdlist mode=2 codepath shares the same wedged static driver objects as mode=1; **W11's PDE shift was specific to mode=0 (regular cmdlists), not a "any non-default cmdlist mode" effect.** The wedge is mode-agnostic across mode={1,2}, only changes when the cmdlist allocation pattern fundamentally changes (mode=0). |
| W12 | `ZE_FLAT_DEVICE_HIERARCHY=COMBINED` (root device + 2 sub-devices, instead of 12 flat tiles). Different per-process driver topology. | **FAIL but DIFFERENT — true HBM OOM**, not UR40. Ranks 1,2,5 SINGLE-BWD FAILED step=0 with `OutOfMemoryError: Tried to allocate 2.32 GiB. GPU N has total capacity of 63.98 GiB of which 2.28 GiB is free. Of allocated memory 29.93 GiB is allocated by PyTorch.` (log `ray_colo_logs/W12_combined_hierarchy_20260507_230010/run.log`). Gen completes cleanly first (success 0.78). | **Configuration mismatch under COMBINED.** PyTorch alloc grew 5× (≈6 GiB → 29.93 GiB) because the trainer sees fewer/larger devices in COMBINED mode and over-allocates per device. KV budget also grew (18.2 → 19.2 GiB). Not a clean diagnostic of the wedged resource — the under-the-hood L0 footprint is invisible inside this real OOM. **What W12 does prove**: the failure is path-dependent on driver topology, the wedged static footprint cannot simply be sized in HBM by switching topologies, and FLAT vs COMBINED produce architecturally different allocation patterns at PyTorch level. |
| W17 | **Sequential co-tenancy** — `TORCHTUNE_RAY_COLOCATE_KILL_AFTER_GEN=1`. After generation succeeds and qr is broadcast to all trainer ranks, rank 0 drops `self._vllm_llm` and calls `ray.shutdown()`. All ranks then sleep `TORCHTUNE_RAY_COLOCATE_DRAIN_S` seconds (default 5) so i915 + L0 reclaim per-client per-tile state. Then trainer ranks enter BWD. Recipe hook at `recipes/dev/grpo_full_finetune_distributed_xpu.py:2807` (gated by env var). | **PASS through BWD + optimizer step on all 8 ranks** (log `ray_colo_logs/W17_kill_after_gen_20260507_231104/run.log`). All ranks log `optimizer done` at 23:15:10. The teardown took 1.42s; the step crashes only at the *next* step's `_sync_ray_colocate_weights` call because the recipe still references `self._vllm_llm.llm_engine` after we set it to None — which is the expected next-iteration error, not the BWD wedge. | **THE BREAKTHROUGH.** The wedge is from **live concurrent co-tenancy**, not from sticky per-process driver state. Killing the vLLM driver client BEFORE the trainer's first BWD chunk unblocks BWD on the same tiles, in the same trainer process, with the same per-tile L0 driver state that has been wedging for every prior W-probe. **Implications**: (a) every prior W-probe (W4-W16) was attacking the wrong axis — none of them changed the live-co-tenancy state, so none of them moved the symptom. (b) The wedged resource is held by the *combined* event/queue/cmdlist activity of two L0 clients executing on the same tile concurrently, not by any per-client static footprint. (c) A practical workaround exists: kill+respawn vLLM around every BWD. (d) Production fix W2 still preferred for throughput; W17/W19 unblocks fully-colocated TP=8 if HBM is the binding constraint. |
| W18 | **Naive kill+respawn cycle** — same W17 hook on gen-side, plus rank-0 calls `_init_vllm_ray_colocate(self._cfg)` post-BWD/opt to rebuild vLLM before next-step gen. Trainer ranks 1-7 wait on the default `torch.distributed.barrier()` (XCCL on XPU). Recipe hook at line 4150. | **HUNG at vLLM respawn.** Step 0 BWD/opt cleared cleanly per W17 (all 8 ranks logged `optimizer done`). Then rank 0 began `LLM(...)` and never returned. py-spy on the Ray vLLM workers showed them stuck in `vllm/v1/worker/xpu_worker.py:165 init_device → all_reduce` while trainer ranks 1-7 sat in xccl `barrier()` on the same 8 tiles. (log `ray_colo_logs/W18_kill_respawn_20260507_234254/run.log`). | **Same wedge, rotated 180°.** W17 proved live concurrent L0 co-tenancy is the wedged axis. W18 then re-introduced it from the OTHER direction — vLLM TP-init posting `all_reduce` while trainer barrier-ops were already live. Two L0 clients × same 8 tiles = deadlock again. The fix is to make the trainer-side wait XPU-free during respawn (W19). |
| W19 | **Kill+respawn with gloo barrier** — same W17 + W18 hook, but the post-respawn trainer barrier uses `self._ray_colocate_gen_pg` (gloo, CPU-only) instead of the default xccl backend. Ranks 1-7 hold ZERO L0 collectives during rank 0's `LLM(...)` call. Recipe hook at `recipes/dev/grpo_full_finetune_distributed_xpu.py:4168`. | **PASS 3/3 steps**. Steady-state ~290s/step (gen 148s + grpo 16s + kill 1.3s + drain 5s + respawn 30s + wsync 90s). All cycles clean: ratios=1.0000, kl ∈ {0.0008, 0.0007, 0.0031}, no UR40, no banned:1. Final METRICS step=3 emitted before SIGTERM from hold expiry. (log `ray_colo_logs/W19_kill_respawn_20260508_001441/run.log`). | **W17+W19 = the validated fully-colocated TP=8 workaround.** Cost is +17% over the W2 process-isolated baseline (290s vs 247s/step on Qwen3-8B G=4), well below the 3-5× I forecasted. The wsync into freshly-spawned actors is the dominant non-gen cost (~90s), not the LLM cold-start (~30s on warm cache). Use this path when HBM forces full 8-tile colocation; use W2 when 4 spare tiles are available. |

### Finding chain (2026-05-07)

```
W4 fail  → empty_cache cannot drain enough L0 events
W5 fail  → CCS partitioning is wrong resource
SKIPGEN fail → wedge is from actor existence, not generate activity
              → W1 (sleep/wake) cannot help (driver context is process-bound)
W6 fail  → NEO counter-based events stack does not move the symptom
              → wedged resource is not the L0 event pool; ALCF Bug 106 unlikely match
W7 N/A   → MakeEachAllocationResident=1 OOMs vLLM init before reaching BWD
W8 fail  → IPC backing change (pid+fd vs sockets) does not move the symptom
              → wedged resource is not the IPC handle table either
W9 fail  → UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4096 (16× default) does not move
              → conclusively rules out event-pool count (W6 ruled out mechanism, W9 ruled out count)
W10 fail → MakeEachEnqueueBlocking=1 (zero outstanding submissions) does not move
              → rules out dynamic accumulation entirely
              → wedge is from STATIC process-context footprint of two L0 clients on a tile
              → no application-level pacing/batching/throttling can fix this
W11 fail → UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 + UR_L0_USE_DRIVER_INORDER_LISTS=1
              → FAULT MODE SHIFTS: not UR40, but i915 GPU PDE (CCS NotPresent PML4 Atomic, banned:1)
              → command-list mode / objects ARE implicated in the static driver footprint
              → first positive signal differentiating driver-side resource classes
              → regular + driver-inorder path exposes a different driver bug, not a fix
W15 fail → ZEX_NUMBER_OF_CCS=4 on ALL 12 tiles (W5 was misconfigured; only set tile 0)
              → bit-identical UR40 step-0 BWD signature
              → rules out CCS-mode-mismatch as the wedge mechanism (Intel doc said
                mismatch causes submission stall; correctly matched modes do not help)
W16 fail → UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH=4096 + CLEANUP_THRESHOLD=1 + BATCH_SIZE=1
              → bit-identical UR40 step-0 BWD signature
              → lifting cmdlist-pool ceilings does NOT lift the wedge (cf W9 for events)
              → wedged resource is in EXISTENCE of cmdlist family, not its capacity
              → candidates: cmdlist allocator state, persistent submission bufs,
                KMD-side cmdlist bookkeeping, fixed-size scratch reservations
W13 fail → CCL_SYCL_*_TMP_BUF=1 (persistent oneCCL temp buffers, bypass per-call IPC)
              → bit-identical UR40 step-0 BWD signature
              → wedge is BELOW oneCCL — in the L0 driver layer directly
              → IPC handle table ruled out at TWO layers (UR via W8, oneCCL via W13)
W17 PASS → kill vLLM driver + ray.shutdown() AFTER gen, drain 5s, THEN BWD
              → ALL 8 ranks complete BWD + optimizer step cleanly
              → ★ MECHANISM: wedge is from LIVE concurrent co-tenancy
              → wedged resource is the *interaction* of two L0 clients on a tile,
                NOT any per-client static footprint
              → every prior W4-W16 left actors LIVE during BWD → that's why none moved the symptom
              → application-level workaround IS viable: kill+respawn vLLM around each BWD
W2 pass  → process isolation (8+4); preferred for throughput when HBM allows

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
- (internal investigation history retained in the project's working notes).

## Status

**ROOT CAUSE CONFIRMED; mitigation sweep complete (2026-05-08).** The wedge is the static
process-context footprint of two L0 driver clients sharing a tile (W4–W16 all negative; W10
rules out dynamic accumulation; W17 isolates live co-tenancy as the mechanism). Two validated
production paths: **W2** (8+4 process isolation, 247s/step Qwen3-8B) and **W17+W19** (kill+respawn
around each BWD, 290s/step). What remains is the Intel-side ask (see "What we need from Intel"):
a per-tile L0 resource ceiling that two co-resident clients cannot both satisfy during an FSDP
backward. No application-level fix is possible without process isolation or a driver change.
