# Aurora L0/UR/oneCCL resource-lifetime signatures — four reproducible failures, possibly related

**Target**: github.com/argonne-lcf/AuroraBugTracking (cc [email protected])
**Filer**: ngetty / ALCF ModCon (TorchTune RL on Aurora)
**Frameworks tested**: `frameworks/2025.3.1` (PyTorch 2.10.0a0+git449b176, oneCCL 2021.17, Level Zero 1.24.0, I915_25.2.29). Persists on torch 2.11.0+xpu against the same `oneapi/release/2025.3.1` module.
**Possibly related upstream ticket**: ALCF AuroraBugTracking #106 / GSD-12152 (`zeEventPoolDestroy` cross-pool dependency hang) — but our W6 result (counter-based events) suggests this is **not** the same mechanism for at least one of our signatures.

## TL;DR

Four reproducible failure signatures on Aurora that all manifest as either `UR_RESULT_ERROR_OUT_OF_RESOURCES` (= L0 error code 40) or as `banned:1` PDE/PML5 page faults. They likely share a Level Zero / Unified Runtime / oneCCL resource-lifetime *family* but the evidence does not yet identify a single shared pool. **For signature #1 specifically, our experiments W4/W5/W6/W8/W9/W10/W11/W11a/W13/W15/W16/SKIPGEN have ruled out: empty_cache drains, CCS partitioning (correctly applied via the documented per-root-device PVC syntax to all 12 tiles), the L0 event-pool mechanism, the L0 event-pool count (16×), the L0 IPC handle backing, the oneCCL temp-buffer path, the immediate-cmdlist pool ceilings, the immediate-cmdlist mode ({1,2}), generation activity, and dynamic outstanding-submission accumulation.** What remains is the static per-process driver footprint of two L0 clients sharing a tile. **Only W11 (`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` + `UR_L0_USE_DRIVER_INORDER_LISTS=1`) shifts the failure class away from UR40 — to a GPU PDE (`ctx_id:1 (CCS) NotPresent PML4 Atomic, banned:1`) — implicating the command-list path as part of the wedged static footprint, but the alternative cmdlist mode (W11a, mode=2) reverts to UR40, so cmdlist-mode tuning is not a workaround.** We are asking Intel/ALCF for help localizing the wedged resource and for guidance on the public UR-L0 and oneCCL knobs we have not yet exhausted.

## The four signatures

| # | Signature | Trigger | Crash point | Doc |
|---|-----------|---------|-------------|-----|
| 1 | UR 40 at step-0 BWD under per-tile process co-tenancy | Two OS processes attached to same PVC tile both submit L0 work — even when the second process never calls `vllm.generate()` (SKIPGEN). Actor *existence* (L0 attach + KV pre-alloc) is sufficient. | First BWD chunk of step 0; ~30 GiB HBM headroom | `docs/bugs/xpu_l0_event_pool_co_tenancy.md` |
| 2 | UR 40 from `empty_cache()` + FSDP `storage.resize_()` cycle | `torch.xpu.empty_cache()` between FSDP forwards | Iter 70 (FSDP2) / iter 145 (FSDP1), bit-deterministic | `docs/bugs/intel_xpu_resource_leak_bug_report.md` |
| 3 | banned:1 PDE — oneCCL L0 IPC handle cache | LRU eviction at default threshold (~step 28) → stale-VA AllGather; or accumulation at threshold=65536 → 10.85 GiB metadata OOM | Step 28 dense-32B / step 2 EP=16 | `docs/bugs/ccl_ipc_handle_cache.md` |
| 4 | banned:1 PML5 — `XPUPluggableAllocator` + FSDP cross-stream VA churn | Any custom allocator registered via `XPUPluggableAllocator` | Step 1 compute, NOT during a CCL collective. **L0 IPC was ruled out by `LD_PRELOAD` shim tracing — zero `zeMemGetIpcHandle` calls fire on this path.** | `docs/bugs/xpu_pluggable_allocator_record_stream.md` |

All four have deterministic reproducers and ruled-out alternatives in the linked files.

## What we believe and what we don't

**Believe**: signatures #1-#4 all involve per-process / per-tile L0 or UR object lifetime. UR 40 = `UR_RESULT_ERROR_OUT_OF_RESOURCES` is an accounting failure inside the per-tile L0 driver state. The `banned:1` faults are *consistent with* stale / freed / unresident VA access — same family, different concrete mechanism.

**Don't believe (yet)**: that they are all the *same* pool. W6 evidence (counter-based events did not move signature #1) argues that signature #1 is not event-pool-bound. Signature #4 has `LD_PRELOAD` evidence ruling out L0 IPC, so it is not the same mechanism as signature #3.

**Strongest cross-framework signal**: surveyed NeMo-RL, torchrl, OpenRLHF — all use Ray-actor + vLLM patterns; none reports an analog of #1. CUDA stacks have no per-tile event/queue pool ceiling and no IPC handle cache eviction race. So signatures #1 and #3 are Aurora-stack-specific.

## Local experiments already run on signature #1

Documented full chain in `docs/bugs/xpu_l0_event_pool_co_tenancy.md`; condensed:

| ID | Lever | Result |
|----|-------|--------|
| W4 | `torch.xpu.empty_cache()` between gen and bwd | FAIL — UR40 step=0 |
| W5 | `ZEX_NUMBER_OF_CCS=0:2` (CCS partitioning) | FAIL — UR40 step=0 |
| SKIPGEN | Ray actors exist but `.generate()` is bypassed | **FAIL — UR40 step=0**. Actor existence alone wedges BWD. Generation activity is irrelevant. |
| W6 | `NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=1 ForceInOrderImmediateCmdListExecution=1 EnableTimestampPoolAllocator=1`. Driver confirms knobs honored via `PrintDebugSettings=1`. | **FAIL — UR40 step=0**. Counter-based events do not move the symptom. **Rules out the L0 event-pool *mechanism*.** |
| W8 | `EnablePidFdOrSocketsForIpc=0` (force pid+fd L0 IPC backing instead of socket-based) | **FAIL — UR40 step=0**. **Rules out the L0 IPC handle table.** |
| W9 | `UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4096` (16× default 256) | **FAIL — UR40 step=0**. **Rules out the L0 event-pool *count*.** |
| W10 | `MakeEachEnqueueBlocking=1` (zero outstanding kernel/transfer submissions per process at any moment) | **FAIL — UR40 step=0**. NEO knob confirmed honored. **Rules out dynamic accumulation entirely.** |
| W11 | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` + `UR_L0_USE_DRIVER_INORDER_LISTS=1` (regular command lists with driver-side in-order semantics, instead of immediate command lists) | **FAIL but FAULT MODE SHIFTS — not UR40.** Step-0 BWD fails with `Segmentation fault from GPU at 0x0, ctx_id:1 (CCS) NotPresent PML4 Atomic, banned:1` followed by NEO i915 `drm_neo.cpp:288` abort + SIGABRT. **First probe to change failure class. Implicates command-list mode in the static driver footprint, but the alternative mode is also broken under co-tenancy.** |
| W11a | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=2` (per-thread immediate cmdlists, ostensibly best isolation) | **FAIL — bit-identical UR40 step=0**. Per-thread cmdlist isolation does not help. W11's PDE was specific to mode=0 + driver in-order; mode=2 reverts to the baseline UR40 wedge. **Rules out cmdlist-mode tuning as a workaround on either path.** |
| W12 | `ZE_FLAT_DEVICE_HIERARCHY=COMBINED` (consolidate 2 tiles per root → 6 root devices instead of 12) | **FAIL — DIFFERENT failure class: real HBM OOM**. Trainer alloc grew from ~6 GiB/tile to 29.93 GiB/tile under COMBINED (driver topology mismatch with the recipe's per-tile sizing). Not a clean diagnostic — just shows the failure is path-dependent on driver topology and the wedged static footprint cannot simply be sized away by switching topologies. |
| W13 | `CCL_SYCL_*_TMP_BUF=1` for allgatherv/reduce_scatter/allreduce/broadcast/reduce (oneCCL persistent device-side temp buffers, bypasses per-call L0 IPC handle exchange) | **FAIL — bit-identical UR40 step=0**. **Rules out the oneCCL temp-buffer path** as the wedge — failure is below oneCCL, in the L0 driver state directly. |
| W15 | `ZEX_NUMBER_OF_CCS="0:4,1:4,...,11:4"` — CCS=4 on ALL 12 tiles via correct PVC syntax (W5 was misconfigured per intel/compute-runtime MULTI_CCS_MODES.md and only set tile 0). Both trainer process AND ray-start AND all Ray actors inherit it. | **FAIL — bit-identical UR40 step=0**. CCS partitioning, even when applied to every tile and ensuring matching modes between co-tenant processes, does not move the wedge. **Rules out compute-context partitioning as a workaround.** |
| W16 | `UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH=4096` (16× default) + `UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=1` (force aggressive cleanup) + `UR_L0_BATCH_SIZE=1` (minimum batch size) | **FAIL — bit-identical UR40 step=0**. **Rules out the immediate-cmdlist pool ceilings.** The wedged resource is below the pool — either the cmdlist allocator state itself, persistent submission buffers, or driver-side per-client cmdlist bookkeeping. |
| W2 | Process-isolated tiles (8+4) | **PASS — production**. 10/10 clean steps, exit=0. |

The SKIPGEN finding combined with W10 is the strongest upstream-routable signal: **actor existence is sufficient (SKIPGEN), and zero outstanding submissions does not help (W10)**. This is not a generation-time / kernel-time / collective-time bug. It is not a dynamic-accumulation bug. **It is a process-attach-time / static-resource bug — the per-tile L0 driver state simply cannot tolerate two co-resident clients running an FSDP backward.**

The wedged resource is *neither* the L0 event pool (W6, W9) *nor* the L0 IPC handle table (W8) *nor* anything dynamic-accumulation-bound (W10). W11 is the first probe to change the failure *class* (UR40 → i915 GPU PDE on a NotPresent PML4 with Atomic access on a CCS context), which **implicates the command-list mode / objects** as part of the wedged static driver footprint. The remaining candidate static bookkeeping objects are: command-list pool, command-queue pool, persistent submission objects, USM context allocator state, KMD-side per-client state, or per-process kernel/SLM scratch reservations.

## Diagnostic asks — split by subsystem

### L0 / Unified Runtime asks (signature #1)

1. **Which L0/UR object counter maps to `UR_RESULT_ERROR_OUT_OF_RESOURCES` in our reproducer, and is that counter per process, per context, per subdevice/tile, or per root device?** This is the single most useful confirmation Intel could provide.
2. Public env knob to query and adjust the **per-tile** ceilings on: events, command-list objects, command-queue objects, submission objects, KMD allocations. Today we have only post-mortem UR 40.
3. Whether signature #1 corresponds to ALCF AuroraBugTracking #106 / GSD-12152 (`zeEventPoolDestroy` cross-pool hang). Our W6 result suggests probably not, but we would like Intel to confirm.
4. Documentation of which L0 resources are per-process vs per-tile vs per-context. Lifetime semantics for `zeEventPool*`, `zeMemOpenIpcHandle` cache, `zeCommandList*` submission objects.

#### Public UR-L0 knobs we plan to test (please confirm honored on Aurora's UR/L0 path, and whether oneCCL bypasses them):

- `UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL` (default 256) — **highest-priority** test for the event-pool hypothesis.
- `UR_L0_DISABLE_EVENTS_CACHING` — distinguish cached-events exhaustion from caching protection.
- `UR_L0_REUSE_DISCARDED_EVENTS=0` — disable reuse of uncompleted in-order-queue events.
- `UR_L0_USE_DRIVER_INORDER_LISTS=1` — pair with counter-based events.
- `UR_L0_USE_IMMEDIATE_COMMANDLISTS={0,1,2}` — co-tenancy may correlate with command-list object count.
- `UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH` (default 256), `UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_MAX` (default 10), `UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD` (default 20) — bound outstanding batches.
- `UR_L0_DEVICE_SCOPE_EVENTS={0,1,2}` — reduce host-visible event pressure.
- `UR_L0_USE_COPY_ENGINE`, `UR_L0_USE_COMPUTE_ENGINE`, `UR_L0_BATCH_SIZE`, `UR_L0_COPY_BATCH_SIZE` — submission-engine pressure diagnostics.
- `UR_L0_SERIALIZE={1,2}` — race-vs-capacity diagnostic.
- `UR_L0_USM_ALLOCATOR_TRACE=1`, `UR_L0_LEAKS_DEBUG=1`, `UR_LOG_LEVEL_ZERO=level:info;flush:warning;output:file,<path>` — for upstream artifacts on the small reproducers.

#### NEO compute-runtime knobs we plan to test (already confirmed `NEOReadDebugKeys=1` is gated on by the i915_25.2.29 driver):

- `MakeEachEnqueueBlocking=1` — race-vs-static-footprint diagnostic. If signature #1 passes with this, outstanding-submission pressure is implicated; if it still fails, static process-context footprint alone is enough.
- `ForceInOrderEvents=1` (distinct from counter-based conversion).
- `ForceImplicitFlush=1` — batched submission object buildup diagnostic.
- `DisableResourceRecycling=1` — distinguish stale reuse vs hard exhaustion.
- `PrintTimestampPacketUsage=1` / `TrackNumCsrClientsOnSyncPoints=1` — confirm whether timestamp/event packets or CSR clients grow before failure.

### oneCCL asks (signature #3)

1. **Reference-count IPC handles by collective** rather than LRU-evict by VA. The current behavior creates a window where one rank evicts a handle while a sibling still has outstanding DMA on it. (See `docs/bugs/ccl_ipc_handle_cache.md` "Recommended upstream fix".)
2. Add `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD` to the documented diagnostic surface alongside `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD`.
3. Document the supported production setting and behavior of `CCL_SYCL_*_TMP_BUF` (persistent temp-buffer collectives that bypass L0 IPC for user buffers). The oneCCL public docs describe these but Aurora-specific guidance is missing. **This is the most plausible non-structural workaround we have not exhausted for signature #3.**
4. Confirm whether `CCL_ZE_IPC_EXCHANGE` valid values are `sockets|pidfd` only on Aurora (`drmfd` is rejected at startup; this should be in the doc).

#### oneCCL knobs we plan to test as workaround/diagnostic for signature #3:

- `CCL_SYCL_ALLGATHERV_TMP_BUF=1`, `CCL_SYCL_ALLREDUCE_TMP_BUF=1`, `CCL_SYCL_BROADCAST_TMP_BUF=1`, `CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1` — bypass L0 IPC for user buffers via persistent temp buffer.
- `CCL_ENABLE_SYCL_KERNELS={0,1}` — path-selection diagnostic.
- `CCL_ALLGATHER=direct`, `CCL_REDUCE_SCATTER=direct`, `CCL_ALLREDUCE=direct` — diagnostic only (non-topo paths copy through host).
- `CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=0`, `CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=0` — copy-engine vs compute-kernel transfer diagnostic.
- `CCL_LOG_LEVEL=info` — capture selected-algorithm tables.
- `CCL_ZE_TMP_BUF_SIZE` — relevant if testing temp-buffer paths.

### PyTorch / XPU allocator asks (signatures #2, #4)

1. Whether `torch.xpu.empty_cache()` can become **allocator-internal for FSDP-owned resized storages**, or whether UR/L0 handle release can be made deterministic after FSDP `storage.resize_()`.
2. A PyTorch XPU allocator **defrag API that does not return segments to Level Zero**. Large models need defrag without the `zeMemFree` cycle.
3. Whether the Python `XPUPluggableAllocator` wrapper can expose `record_stream` hooks. Even if FSDP2 does not use them, the current no-op is a correctness trap for other paths.
4. Whether custom XPU allocators can integrate with oneCCL/OFI/L0 handle invalidation on free.

## Reproducer paths (single-node debug-scaling, 1-hour walltime is enough)

```bash
# Signature 1 (UR40 co-tenancy, TP=8 colocated Ray) — reproduces in <5 min
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/run_qwen3_8b_colocate_ray.sh 3 4

# Signature 2 (empty_cache+FSDP UR40) — reproduces in <2 min
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    /lus/flare/projects/ModCon/ngetty/torchtune/experiments/arena_ipc/repro_xpu_resource_leak.py \
    --fsdp --layers 12 --hidden 1024 --heads 8

# Signature 3 (CCL IPC handle banned:1) — needs ~30 steps, ~30-60 min
# See docs/reports/ccl_external_memory_growth_32b.md

# Signature 4 (pluggable allocator banned:1 PML5) — FSDP2 32B 10 ranks
# See experiments/arena_ipc/diag_findings.md Phase 1b / 2 / 5
```

## What works today (so we are not stuck — but production-suboptimal)

- Signature 1: process-isolated tiles (W2 — trainer 0..7, vLLM TP=4 on 8..11). 10/10 clean steps, exit=0. Sacrifices the Ray-colocate-on-shared-tiles design that NeMo-RL/OpenRLHF/etc. assume.
- Signature 2: never call `torch.xpu.empty_cache()` in FSDP loops. Adequate for ≤32B; **insufficient for 72B** (allocator fragments to OOM without periodic defrag).
- Signature 3: 2-node HSDP for 32B; threshold=65536 trades stale-eviction for accumulation OOM. Still loses single-node 32B.
- Signature 4: avoid `XPUPluggableAllocator` entirely; default allocator with `gc:0.95`. Loses any custom-allocator strategy.

## Doc references (cite in filing)

- L0 counter-based event pools (note: experimental extension page says deprecated, replaced by counter-based events in programming guide): https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_Exp_CounterBasedEventPools.html
- L0 extension registry: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT.html
- UR L0 env vars: https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html
- UR logging/layers: https://oneapi-src.github.io/unified-runtime/core/INTRO.html
- oneCCL env vars: https://uxlfoundation.github.io/oneCCL/env-variables.html
- Intel compute-runtime debug variables source: https://raw.githubusercontent.com/intel/compute-runtime/master/shared/source/debug_settings/debug_variables_base.inl
