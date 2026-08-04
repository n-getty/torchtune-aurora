# ALCF AuroraBugTracking — ready-to-file issue drafts (2026-07-06)

Filer prep for github.com/argonne-lcf/AuroraBugTracking (form:
`issues/new?template=1-BugReportForm.yaml`). Each draft below maps 1:1 to the form fields:
**Point of Contact**, **Contact Details**, **Vendor/ALCF/other tickets**, **Reproducer Path**,
**Status**, **Details**, **Priority**, **ETA**.

Common environment (paste into each Details block):
```
System:      Aurora (ALCF)
Module:      frameworks/2025.3.1   (newest available as of 2026-07-06)
PyTorch:     2.10.0a0+git449b176 (xpu)   [also reproduced on torch 2.11.0+xpu, same module]
oneCCL:      2021.17
Level Zero:  1.24.0.0-i1146
i915:        I915_25.2.29_PSB_250224.35
OS:          SLES 15 SP4, kernel 5.14.21-150400.24.*-default
HW:          Intel Data Center GPU Max 1550 (PVC), 12 tiles/node, 64 GiB HBM/tile
Repo commit: torchtune @ 1823c982 (reproducers self-contained; only torch+CCL needed)
```

Dedup status (searched full tracker 2026-07-06, 72 issues open+closed): drafts #4/#9 = NO MATCH;
draft #5 = RELATED-BUT-DISTINCT from **#143** (cross-referenced in its body). See
`memory/project_alcf_tracker_dedup_20260706.md`.

> **Point of Contact / Contact Details are left as `<TBD>`** — fill with your name + email at submit
> time. Everything else is paste-ready.

---

## DRAFT #4 — oneCCL rejects `expandable_segments` XPU tensors (`invalid usm pointer type: unknown`)

- **Point of Contact:** `<TBD>`
- **Contact Details:** `<TBD>`
- **Vendor/ALCF/other tickets:** (none)
- **Reproducer Path:** `/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/repro_ccl_expandable_segments.py`
- **Status:** Open (WA available)
- **Priority:** (leave unchecked)
- **ETA:** —

> **HW re-verified 2026-07-06** on `frameworks/2025.3.1`, node x4313c2s7b0n0: default allocator →
> `Rank 0/1: allreduce OK, sum=2.0`; `PYTORCH_ALLOC_CONF=expandable_segments:True` →
> `oneCCL: coll_check.cpp:68 ccl_check_usm_pointers: EXCEPTION: coll: allreduce - invalid usm pointer
> type: unknown for device type: gpu` (rc=1). Matched pair, still active.

**Details:**

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` makes **every** oneCCL collective
(allreduce/allgather/reduce_scatter) fail on XPU:
```
coll_check.cpp:68 ccl_check_usm_pointers: condition is_valid_type failed
coll: allreduce - invalid usm pointer type: unknown for device type: gpu
```
PyTorch's `expandable_segments` allocator backs tensors with SYCL `ext_oneapi_virtual_mem`
(L0 `zeVirtualMemReserve`/`zeVirtualMemMap`). Those pointers are valid device memory (compute
kernels work) but `zeMemGetAllocProperties` classifies them `ZE_MEMORY_TYPE_UNKNOWN` (documented L0
behavior for non-USM virtual ranges), and oneCCL's `ccl_check_usm_pointers` rejects anything not
`device`/`host`/`shared`.

There is a **second, deeper** failure past the type check: for tensors > ~8–12 MB oneCCL switches to
the zero-copy IPC path (`zeMemGetIpcHandle`), which does not support `zeVirtualMemMap` ranges and
GPU-page-faults (`banned:1`) even when the type check is bypassed via LD_PRELOAD shim. So production
FSDP2 param tensors (32–256 MB) cannot use expandable_segments under CCL at all.

Steps to reproduce (2 ranks, ~2 min):
```
module load frameworks
# PASS:
ZE_AFFINITY_MASK=0,1 python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  recipes/dev/repro_ccl_expandable_segments.py
# FAIL:
PYTORCH_ALLOC_CONF=expandable_segments:True ZE_AFFINITY_MASK=0,1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  recipes/dev/repro_ccl_expandable_segments.py
```

Impact: `expandable_segments` is the natural fix for the mixed-size fragmentation of RL/FSDP
workloads (measured 7× lower fragmentation overhead at 256 MB allocs), but it is unusable with any
oneCCL-on-CXI process (training FSDP or vLLM TP). Full 2-phase LD_PRELOAD root-cause + suggested
oneCCL fixes (accept virtual-mem device pointers via a secondary L0 query, or provide
`CCL_FORCE_STAGING=1`) in `docs/bugs/intel_ccl_expandable_segments_bug.md`.

Workaround: do not set `expandable_segments:True`; use the default XPU allocator (fragments) or a
custom USM pluggable allocator (has its own separate bug).

---

## ~~DRAFT #5~~ → **DO NOT FILE AS NEW. This is ALCF #143 (MLSL-4397) surfacing via FSDP's all_gather.**

**VERDICT (HW-decided 2026-07-06): #5 is NOT a distinct bug.** The instrumented discriminator (job
8648224, node x4103c2s5b0n0, `experiments/colocate/verify_5_vs_143.sh`, log
`experiments/colocate/bugverify_logs/5disc_leak_20260706_204751.log`) shows the FSDP+empty_cache crash
is a **device-global memory leak with the exact #143 signature**, not the UR-handle exhaustion our doc
claimed:

```
iter  1  torch alloc/resv=1.3/4.6 GiB   ext_free=62.49 GiB
iter 20  torch alloc/resv=1.3/4.6 GiB   ext_free=45.50 GiB
iter 40  torch alloc/resv=1.3/4.6 GiB   ext_free=28.13 GiB
iter 60  torch alloc/resv=1.3/4.6 GiB   ext_free=10.77 GiB
iter 72  torch alloc/resv=1.3/4.6 GiB   ext_free= 0.35 GiB
iter 73  torch alloc/resv=1.3/4.6 GiB   ext_free= 0.08 GiB  -> UR_RESULT_ERROR_OUT_OF_RESOURCES
```

`mem_get_info` free falls **dead-linear at −0.867 GiB/iter** (drops clustered 0.86–0.88) while torch
`memory_allocated`/`memory_reserved` stay pinned at 1.3/4.6 GiB. The crash fires exactly when
device-global memory is **physically exhausted** (~iter 73), NOT at a fixed UR-handle count. This is
#143's mechanism ("PyTorch thinks it released the memory but the device does not get it back;
`mem_get_info` free keeps dropping while torch counters return to baseline") — the same oneCCL leak,
here driven by the collectives FSDP issues (all_gather/reduce_scatter) instead of a bare list-style
`dist.all_gather`.

**This FALSIFIES our own `docs/bugs/intel_xpu_resource_leak_bug_report.md` central claim** that
"memory_allocated is stable throughout — this is NOT a standard memory leak; the leaked resource is UR
handles." We never took the `mem_get_info` reading; when taken, it IS a device-global memory leak,
invisible to torch accounting, class-identical to #143.

**Action instead of filing:** add our FSDP reproducer + this ext_free trajectory as a **comment on
#143 / MLSL-4397**, noting the leak also fires from FSDP's collectives (broader trigger than the
list-style all_gather in the original report) and that the workaround (no empty_cache in FSDP loops) is
the same. Do NOT open a new ALCF issue. `docs/bugs/intel_xpu_resource_leak_bug_report.md` needs its
"UR-handle, not memory" framing corrected (the reproducer, workaround, iteration counts, and 72B impact
all remain valid — only the *named leaked resource* was wrong).

Reproducer: `/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/repro_xpu_resource_leak.py`
(now prints `ext_free`).

<details><summary>Original (now-superseded) draft body — kept for the #143 comment</summary>

**Details:**

On XPU FSDP (FSDP1 and FSDP2), calling `torch.xpu.empty_cache()` between forward passes in an RL
pattern (multiple no-grad forwards + a grad backward) crashes after a **deterministic** iteration
count with:
```
RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
```
(large models instead hit a `banned:1` NotPresent PDE/PML5 fault as the leak outruns the UR counter).

Mechanism: FSDP reshards via `storage.resize_(0)` (returns blocks to PyTorch's caching allocator);
`empty_cache()` then releases those blocks to L0 (`zeMemFree`); the next unshard `storage.resize_(size)`
re-acquires from L0 (`zeMemAllocDevice`) — and **each alloc/free cycle leaks one UR handle**. The leak
rate scales with FSDP-unit count (12L model ~iter 70 FSDP2 / ~iter 145 FSDP1; 72B/81-unit crashes
after ~4 empty_cache calls).

**Isolation (all in the reproducer's flags):** the crash requires ALL THREE of {FSDP + multi-forward
RL pattern + empty_cache between forwards}. Removing any one is stable 200–20000 ops. `memory_stats()`
stays flat throughout — this is an L0/UR-layer leak, invisible to PyTorch accounting.

Steps to reproduce (2 ranks, ~2 min to crash):
```
module load frameworks
# CRASH (~iter 70, FSDP2):
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  recipes/dev/repro_xpu_resource_leak.py --fsdp --layers 12 --hidden 1024 --heads 8
# CRASH (~iter 145, FSDP1):
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  recipes/dev/repro_xpu_resource_leak.py --fsdp1 --layers 12 --hidden 1024 --heads 8
# STABLE (workaround — no empty_cache in chunks):
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  recipes/dev/repro_xpu_resource_leak.py --fsdp --layers 12 --hidden 1024 --heads 8 \
  --no-empty-cache-in-chunks
```

**Relation to #143:** #143 is the list-style `dist.all_gather` **hidden flat temp buffer** not being
reclaimed by empty_cache (device-global free falls, fixed by `all_gather_into_tensor`). THIS bug is
the FSDP `storage.resize_` alloc/free **UR-handle** leak crashing with UR40 — a different code path
and a different leaked object (UR handle, not a CCL temp buffer). Both touch empty_cache on XPU; please
do not close as a dup of #143.

Impact: workaround (never call empty_cache in FSDP loops) is adequate ≤32B but **insufficient for
72B** — without periodic defrag the XPU allocator fragments to OOM (reserved 45.7 GiB vs allocated
24.9 GiB on 48 GiB tiles), and the XPU allocator implements no `garbage_collection_threshold`. Full
analysis + all ruled-out mitigations in `docs/bugs/intel_xpu_resource_leak_bug_report.md`.

Requested fix: release UR handles on `zeMemFree`, OR add a caching-allocator defrag that does not
cycle through L0, OR implement `garbage_collection_threshold` for the XPU allocator.

*(Note: the "UR handle" language above is the superseded framing — see VERDICT at the top of this
section. The real leaked resource is device-global memory, invisible to torch counters, = #143.)*

</details>

---

## DRAFT #9 — Two co-resident L0 clients on one PVC tile → `banned:1` CCS page fault / `UR_RESULT_ERROR_OUT_OF_RESOURCES` at first backward (vLLM+trainer co-tenancy)

- **Point of Contact:** `<TBD>`
- **Contact Details:** `<TBD>`
- **Vendor/ALCF/other tickets:** possibly-adjacent: #106 (`zeEventPoolDestroy` hang), #108
  (immediate-cmdlist hang) — our probes suggest a different mechanism (see ruled-out list)
- **Reproducer Path:** `/lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/pbs_repro_multitile.sh`
  (driver) + `/lus/flare/projects/ModCon/ngetty/torchtune/scratch/repro_colocate_pagefault_multitile.py`
  (torchtune-free; torch+vllm+ipex+mpi4py only)
- **Status:** Open
- **Priority:** (leave unchecked)
- **ETA:** —

> **HW re-verified 2026-07-06** (job 8648191, `qsub -v STEPS=12,LOAD_REAL=1
> experiments/colocate/pbs_repro_multitile.sh`, log `repro_logs/MT_20260706_201909.log`): the 12-tile
> in-process-vLLM + XCCL run with the real Qwen3-4B `load_weights` publish crashed byte-identically —
> `Segmentation fault from GPU at 0xff00ffffffe00000, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 1
> (PDE), access: 0 (Read), banned: 1` (rc=143). Matches the documented 2-factor signature; still
> active on the current module.

**Details:**

When two Level-Zero driver clients attach to the same PVC tile and both submit work, the tile faults
at the **first backward of step 0** with ~24–55 GiB HBM still free (NOT an OOM). Two observed forms,
same family:
```
Segmentation fault from GPU at 0xff00ffff........, ctx_id: 1 (CCS) type: 0 (NotPresent),
  level: 1 (PDE), access: 0 (Read), banned: 1, aborting.        # in-process colocate form
level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)   # Ray 2-process form
```

**Root cause isolated to a 2-factor co-residence interaction** (torchtune-free repro, N=2
byte-identical):

| config | real vLLM `load_weights` into live engine? | concurrent multi-rank XCCL on the tile? | result |
|--------|-------------------------------------------|------------------------------------------|--------|
| single-tile vLLM + load_weights | yes | no  | **clean** |
| 12-tile, load_weights skipped   | no  | yes | **clean** |
| 12-tile + real load_weights     | yes | yes | **CRASH (byte-identical PDE)** |

It is the **static co-tenancy of two L0 clients**, not instantaneous concurrency: a `barrier()`+sync
that removes any in-flight collective around the publish does not help. In the two-process Ray TP=8
variant, the equivalent finding is that **actor existence alone** (L0 attach + KV pre-alloc, with
`generate()` skipped) wedges the first backward, and **zero outstanding submissions**
(`MakeEachEnqueueBlocking=1`) still wedges — so it is a process-attach-time static-footprint fault, not
dynamic accumulation.

Steps to reproduce (single node, ~5 min, PBS job — mpiexec --pmi=pmix needs the PBS process group):
```
qsub -v STEPS=12,LOAD_REAL=1 experiments/colocate/pbs_repro_multitile.sh   # -> CRASH step 0
qsub -v STEPS=12,NO_VLLM=1  experiments/colocate/pbs_repro_multitile.sh    # control: XCCL-only, CLEAN
```

**Ruled out (public knobs, all still wedge — full table in doc):** L0 event pool mechanism
(`EnableImplicitConvertionToCounterBasedEvents=1`) AND count
(`UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4096`); L0 IPC handle backing
(`EnablePidFdOrSocketsForIpc=0`) and oneCCL temp-buf path (`CCL_SYCL_*_TMP_BUF=1`); CCS partitioning
(`ZEX_NUMBER_OF_CCS=4` on all 12 tiles, matched modes); immediate-cmdlist pool ceilings
(`UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH`/`_CLEANUP_THRESHOLD`/`_BATCH_SIZE`); cmdlist mode
{1,2}; `MakeEachEnqueueBlocking=1`. `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 + ..._DRIVER_INORDER_LISTS=1`
is the only knob that shifts the failure *class* (UR40 → a null-ptr atomic CCS PDE), implicating the
command-list family in the static footprint, but is not a fix.

**Ask:** which L0/UR object counter maps to `UR_RESULT_ERROR_OUT_OF_RESOURCES` here, and is it
per-process / per-context / per-tile / per-root-device? Any public knob to query/raise the per-tile
ceiling on command-list / command-queue / submission objects? This blocks fully-colocated (single-tile
vLLM + trainer) RL, which CUDA stacks (NeMo-RL/OpenRLHF/torchrl) do routinely — there is no CUDA
analog of this per-tile L0 ceiling.

Workarounds: process isolation (trainer on tiles 0–7, vLLM TP=4 on 8–11) is clean 10/10 steps; or
kill+respawn the vLLM client around each backward (validated, +17% step time). Full W-probe sweep,
mechanism, and the two-process analogue in `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md`
and `docs/bugs/xpu_l0_event_pool_co_tenancy.md`.

---

## Not filing (this round)

- **#3 CCL IPC-handle cache** (`ccl_ipc_handle_cache.md`): high-confidence root cause, NO tracker
  match — but the minimal FSDP2-only reproducer I wrote (`recipes/dev/repro_ccl_ipc_handle_cache.py`)
  **does NOT cleanly isolate the CCL-internal leak** (HW-tested 2026-07-06, node x4313c2s7b0n0). The
  churn leg (per-layer wrap, 40 steps, threshold=65536) dropped device-global free by 24.7 GiB, but
  that drop is **fully explained by torch `reserved` growing 24→54 GiB** to cover the largest seqlen —
  `drop/step` fell to ~0.000 after step 10 (plateau, no steady per-step CCL growth), and the
  `--stable-va` control was **totally flat** (ext_free 41.97 GiB from step 5). **The flaw is
  intrinsic:** the VA churn that fills the IPC-handle cache is the *same* churn that grows the torch
  allocator's reserved floor, so a small model cannot separate "CCL handle metadata" from "allocator
  reserved for the biggest shape seen." The documented ~10 MB/step CCL-internal signature was measured
  in the **full 32B GRPO recipe at steady-state over ~28 steps** where reserved is genuinely flat. To
  file #3 with a minimal repro, either (a) run at real 32B scale with fixed shape for many steps
  (reserved flat, watch for slow ext_free creep), or (b) instrument oneCCL directly (UR_L0 logging /
  handle count) rather than inferring from `mem_get_info`. **Do NOT file #3 on the current minimal
  repro** — it would not survive review. Logs: `experiments/colocate/bugverify_logs/3{a,b}_*.log`.
  **When filed, cross-reference #143 / MLSL-4397 and ask Intel to confirm whether MLSL-4397's oneCCL
  memory-lifetime fix covers the IPC-handle-cache path (`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD`) or
  whether that is a separate oneCCL data structure.** #143 is the all_gather hidden-temp-buffer; #3 is
  the VA-keyed IPC handle table — different mechanism, but both are "oneCCL doesn't reclaim device
  memory," and since #143 was filed on our behalf and routed generically to oneCCL, Intel may treat
  them as one internal bug. Let them decide rather than over-claiming independence.
- **#1 accelerate FSDP host-fallback** (`accelerate_xccl_fsdp_topology_host_fallback.md`): **REFUTED
  as a platform bug** 2026-07-06 — native torchtune FSDP is 2.84× faster on the same node with the
  same `node_dev_uuids` warning, so the warning is benign and the gap is inside HF-accelerate. Not an
  ALCF ticket.
- **#2 FSDP AllReduce deadlock @192 ranks** (`fsdp_gradient_allreduce_deadlock_192rank.md`): sibling
  vjepa2 project, has its own ALCF ticket draft; needs the rank-locality breakdown before filing.
- **#6 IPEX bgmv LoRA PDE** (`ipex_bgmv_lora_pde.md`): file to **IPEX**, not ALCF; do the vLLM-main +
  IPEX-nightly re-check first (not run this session).
- **#7 vLLM PP>1** (`vllm_xpu_pp_kvcache_init.md`): file to **vLLM**, not ALCF.
- **#11 XPUPluggableAllocator record_stream** (`xpu_pluggable_allocator_record_stream.md`): the clean
  sub-bug (Python wrapper never wires `set_record_stream_fn`) is a **PyTorch** issue; the FSDP2 race
  theory is unsettled. File the wrapper no-op upstream to PyTorch, not ALCF.
