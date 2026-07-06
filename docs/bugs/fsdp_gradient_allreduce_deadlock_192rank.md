# Bug: FSDP cross-node gradient-AllReduce deadlock at 192 ranks (silent hang, no crash)

**System:** Aurora (ALCF), Intel Max 1550 / PVC, 12 tiles/node, native `xccl`, OFI transport
(`CCL_PROCESS_LAUNCHER=none`, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=cxi`).
**Scale:** 16 nodes × 12 tiles = 192 ranks. FSDP1 HYBRID_SHARD_ZERO2 (shard within node, replicate across nodes).
**Workload:** NOT this repo — V-JEPA 2.1 ViT-G 2B continued-pretrain (sibling project
`/flare/ModCon/ngetty/vjepa2`). Recorded here because it is the same *outward class* as our
large-scale FSDP hangs and gets conflated with them.
**Status:** open ALCF ticket (draft: `/flare/ModCon/ngetty/vjepa2/docs/ALCF_TICKET_fsdp_allreduce_deadlock.md`),
root cause not closed. Reproduced 2× with captured tracebacks (jobs 8645821, 8646258).

## Symptom

Intermittent (~1 deadlock / 3.5 h) silent hang in the **FSDP backward gradient AllReduce**. Not a
crash, not a dead rank — a collective-boundary split: the cohort desynchronizes across an FSDP
collective and blocks forever. Captured via per-rank `faulthandler.dump_traceback_later`:
- majority (114 / 372 dumps) parked in backward `_reduce_grad` → `all_reduce` over the 16-node
  replicate group (`_runtime_utils.py:872 _reduce_grad` ← `:766 _post_backward_hook`);
- minority (~10 / 24) one collective away in forward unshard (`:421 _pre_forward_unshard`).

2B (depth 48) hangs; identical stack/launcher/config on **1B (depth 40) is clean >11k iters**.
Ruled out by the ticket (with evidence): dead rank, walltime, AllReduce algorithm (ring == double_tree),
memory/IPC leak (flat backward floor), `libpil4dfs`/DAOS-17499 (on Lustre), specific bad node.

## Why this is DISTINCT from every bug in this dir

| Our bug | Discriminator it fails |
|---|---|
| `clip_grad_norm` deadlock (`memory/bugs/project_clip_grad_norm_deadlock.md`, `_v2`) | Needs **vLLM colocated**; hang is in `clip_grad_norm_` all_reduce at `optimizer.step()`. This has no vLLM and blocks in FSDP's own `_reduce_grad`. |
| `xccl_teardown_hang` (`memory/bugs/project_xccl_teardown_hang.md`) | Hang at **process exit** `destroy_process_group()`, tied to vLLM weight sync. This is mid-training, step-level. |
| `wsync_intra_pg_deadlock` (`memory/bugs/project_wsync_intra_pg_deadlock.md`) | DP>1 **vLLM weight-sync broadcast**, not an FSDP gradient collective. |
| `accelerate_xccl_fsdp_topology_host_fallback.md` | A **throughput** bug (steady 13.6 s/step, host-fallback), not a hang; HF-accelerate not native FSDP. Closest *class* only. |
| `ccl_ipc_handle_cache.md`, `intel_ccl_expandable_segments_bug.md`, `xpu_pluggable_allocator_record_stream.md`, `project_step3_bwd_spike.md` | All end in a **`banned:1` GPU page-fault crash**. This is a hang; the ticket rules out the memory-leak class (flat floor). |
| `intel_xpu_resource_leak_bug_report.md` | UR-handle leak → `OUT_OF_RESOURCES` **crash** ~iter 70. Crash, not hang. |

## Caveat on the ticket's framing (before trusting "CXI contention → straggler")

The ticket attributes the hang to transient fabric contention producing a straggler that never
rejoins. That is a plausible *trigger* but is in tension with the *mechanism* the traceback shows:
a pure fabric slowdown on a correct SPMD program does **not** deadlock — all ranks wait at the *same*
collective and drain when the straggler arrives. A **collective-boundary split** (some ranks a
collective *ahead* in forward unshard while the majority is behind in backward reduce) is the
signature of a **rank-schedule divergence** — mismatched collectives that can never rendezvous.
Fabric contention likely just widens the AllReduce window enough for an FSDP1 prefetch/reshard
ordering divergence (or a rank-subset branch, e.g. NaN/inf guard) to manifest.

**Consequence for the ask:** a CCL timeout+retry only helps if the stuck ranks are on the *same*
collective (same PG + shape). If they are on *different* collectives, no CCL tuning recovers it and
the fix is on the FSDP-schedule side. Confirm the two frame groups are issuing the same collective
before promising a CCL-tuning fix will land.

## Cross-reference

Full ticket + per-incident nodefiles + stack dumps:
`/flare/ModCon/ngetty/vjepa2/docs/ALCF_TICKET_fsdp_allreduce_deadlock.md` and
`.../checkpoints/surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak/hang_diag/`.
Same silent-hang *class* other Aurora large-scale jobs report (AGPT/torchtitan 256N
"blind-rotate a node"); this one has a captured traceback.
