# Bug: HF-accelerate FSDP on Aurora falls back to host-staged collectives (oneCCL can't see intra-node topology)

**System:** Aurora (ALCF), Intel Max 1550, `frameworks/2025.3.1` (torch 2.10+xpu, oneCCL 2021.17),
HF `accelerate 1.12.0` + `trl 1.5.1` FSDP (NOT torchtune's native FSDP).
**Status:** root cause narrowed, not closed. Reproduced on Qwen3-4B, 12 tiles, 1 node.

## Symptom

Full-parameter FSDP (`FULL_SHARD`) of Qwen3-4B on one Aurora node runs at **13.6–14.5 s/step,
MFU ~1.7%**. LoRA on the identical setup is fine (2.46 s/step). Decomposing with LoRA as a
compute anchor (same fwd/bwd, trivial comm):
```
full 14.55s − LoRA 2.46s ≈ 12.1 s  = pure FSDP collective = 83% of the step
effective collective BW = ~24 GB/step ÷ 13.6 s ≈ 1.75 GB/s   (host-staged TCP class)
```
1.75 GB/s is neither XeLink (100s GB/s) nor HSN RDMA (~25 GB/s/NIC) → the per-step all-gather is
on a host fallback path.

## Root cause (narrowed)

oneCCL cannot determine that the 12 ranks share a node, so it never builds the XeLink peer group:
```
|CCL_WARN| comm_dev_uuids is not sub-vector of node_dev_uuids, comm_dev_uuids size 12,
          node_dev_uuids size 1, this may happen due to narrow device affinity mask
```
With per-rank `ZE_AFFINITY_MASK=$LOCAL_RANK`, each process exposes exactly one device UUID; CCL's
topology discovery sees `node_dev_uuids size 1` and gives up on intra-node peer mapping.

## What does NOT fix it (verified)

- **CCL collective algorithm** `CCL_ALLGATHER`/`CCL_REDUCE_SCATTER` ring→direct: no change
  (env confirmed applied). Not an algorithm problem.
- **Device visibility alone**: exposing all 12 tiles FLAT (`ZE_FLAT_DEVICE_HIERARCHY=FLAT`, no
  mask) avoids an accelerate "device index out of range" crash but still host-fallbacks; per-rank
  masking fixes `visible_devices_per_rank` (12→1) but triggers the `node_dev_uuids size 1` warning
  above and is *still* 14.5 s. Both extremes break topology discovery, oppositely.
- **`CCL_WORKER_COUNT` (verified 2026-06-19)**: the harness originally hardcoded `=4` (a real
  misconfig — the repo-documented safe value is `1`; `=4` causes a ~48× AllGather regression in the
  production multi-node recipe). Re-running full-FT at `=1` left the step time **unchanged
  (~14.2 s)** and the `node_dev_uuids size 1` warning still fires. So on this mpiexec+accelerate
  path WORKER_COUNT is **not** the bottleneck — the host-fallback collective swamps any
  worker-thread effect. Fix WORKER_COUNT for hygiene, but it does not move this number. (Log:
  `logs/fix2_aurora_full.log`.)

## Likely fix directions (untested)

- Use **torchtune native FSDP** (`recipes/full_finetune_distributed.py`) with the validated xccl
  init in `src/training/distributed.py` (which keeps per-rank mask for native mode and omits
  `device_id` for xccl). If native is compute-bound, the bug is purely accelerate's PG/topology
  setup. **Most likely to yield a correct number fast.**
- Under accelerate, give CCL correct topology without 1-UUID-per-rank: all-tiles-visible +
  explicit per-rank `device_id`/device-mesh so accelerate targets the right tile while CCL still
  sees all node UUIDs. Mind the DataLoader-fork deadlock that `distributed.py` warns about when
  `device_id` is passed under xccl.

## Cross-reference

Same outward class as the **2026-06-17 gloo-CPU-bounce incident** in `docs/RESULTS_DISCIPLINE.md`
(`memory/project_lora_vs_fullft_4b_parity_20260617.md`): comm-heavy dense leg on a host path,
LoRA leg unaffected, CPU tests green. Different code path (that was torchtune `CHUNKED_BACKWARD`
reduce_scatter; this is HF-accelerate FSDP all-gather) but identical failure signature — check
whether a shared gloo/host-bounce mechanism underlies both.

Related CCL/XPU bugs in this dir: `ccl_ipc_handle_cache.md` (L0 IPC handle XeLink peer access — the
fast path this bug fails to establish), `intel_ccl_expandable_segments_bug.md`,
`xpu_l0_event_pool_co_tenancy.md`.

Not the same bug (same fabric/CCL *class*, different failure mode):
`fsdp_gradient_allreduce_deadlock_192rank.md` — a silent FSDP gradient-AllReduce **hang** at 192
ranks (native FSDP, no vLLM), vs. this steady-state **throughput** host-fallback under HF-accelerate.

## Full investigation + repro

`docs/reports/sft_throughput_aurora_vs_polaris_handoff_20260619.md` and harness
`benchmarks/sft_throughput_aurora_vs_polaris/` (logs with the raw CCL_WARN: `logs/sft_bench_aurora.o8549714`).
