# Bug: FSDP cross-node gradient-AllReduce deadlock at 192 ranks (silent hang, no crash)

**System:** Aurora (ALCF), Intel Max 1550 / PVC, 12 tiles/node, native `xccl`, OFI transport
(`CCL_PROCESS_LAUNCHER=none`, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=cxi`).
**Scale:** 16 nodes × 12 tiles = 192 ranks. FSDP1 HYBRID_SHARD_ZERO2 (shard within node, replicate across nodes).
**Workload:** NOT this repo — V-JEPA 2.1 ViT-G 2B continued-pretrain (sibling project
`/flare/ModCon/ngetty/vjepa2`). Recorded here because it is the same *outward class* as our
large-scale FSDP hangs and gets conflated with them.
**Status:** open ALCF ticket (draft: `/flare/ModCon/ngetty/vjepa2/docs/ALCF_TICKET_fsdp_allreduce_deadlock.md`,
updated 2026-07-06), root cause not closed. Reproduced 2× with captured tracebacks (jobs 8645821, 8646258).

## Symptom

Silent hang in the **FSDP backward gradient AllReduce** at a **measured MTTF of ~3h43m** (walltime
set to 4 h in-script since MTTF < walltime → no job reaches the wall; training limps forward
e15→e214+/~epoch 313 via self-healing checkpoint-resume). Not a crash, not a dead rank — a
collective-boundary split: the cohort desynchronizes across an FSDP collective and blocks forever.
Captured via per-rank `faulthandler.dump_traceback_later`:
- majority (114 / 372 dumps) parked in backward `_reduce_grad` → cross-node `all_reduce` over the
  16-node replicate PG (`_runtime_utils.py:872 _reduce_grad` ← `:766 _post_backward_hook`);
- minority (~10 / 24) in **forward** `_pre_forward_unshard` (`:421`) — a within-node `all_gather` on
  the shard PG. NB HYBRID_SHARD_ZERO2 = SHARD_GRAD_OP, so there is no backward re-unshard: those
  ranks are in a genuine forward pass, i.e. a full iteration skewed from the backward majority.

2B (depth 48) hangs; identical stack/launcher/config on **1B (depth 40) is clean >11k iters**
(2B has ~2× the per-rank AllReduce payload — consistent with "longer window → more chance for a
schedule skew to open," not necessarily "more bytes → more fabric stress").
Ruled out by the ticket (with evidence): dead rank, walltime, AllReduce algorithm (ring == double_tree),
memory/IPC leak (flat backward floor), `libpil4dfs`/DAOS-17499 (on Lustre), specific bad node.

**Lag cause staged as hypotheses (ticket 2026-07-06 is honest that this is unsettled):**
(1) transient CXI/dragonfly fabric contention (leading, not isolated); (2) **decode-latency
outliers** — the captured-hang jobs ran `num_workers=2` with a high-bitrate source (heichole,
~1765 ms/clip, ~4× others); at `num_workers=0` dataload-time is p50≈1s/p99≈10s with decode on the
critical path. A clean isolation is running (nw=0 + heichole re-encoded); if deadlocks persist at the
same MTTF that isolates the fabric hypothesis.

## Why this is DISTINCT from every bug in this dir

| Our bug | Discriminator it fails |
|---|---|
| `clip_grad_norm` deadlock (`memory/bugs/project_clip_grad_norm_deadlock.md`, `_v2`) | Needs **vLLM colocated**; hang is in `clip_grad_norm_` all_reduce at `optimizer.step()`. This has no vLLM and blocks in FSDP's own `_reduce_grad`. |
| `xccl_teardown_hang` (`memory/bugs/project_xccl_teardown_hang.md`) | Hang at **process exit** `destroy_process_group()`, tied to vLLM weight sync. This is mid-training, step-level. |
| `wsync_intra_pg_deadlock` (`memory/bugs/project_wsync_intra_pg_deadlock.md`) | DP>1 **vLLM weight-sync broadcast**, not an FSDP gradient collective. |
| `accelerate_xccl_fsdp_topology_host_fallback.md` | A **throughput** bug (steady 13.6 s/step, host-fallback), not a hang; HF-accelerate not native FSDP. Closest *class* only. |
| `ccl_ipc_handle_cache.md`, `intel_ccl_expandable_segments_bug.md`, `xpu_pluggable_allocator_record_stream.md`, `project_step3_bwd_spike.md` | All end in a **`banned:1` GPU page-fault crash**. This is a hang; the ticket rules out the memory-leak class (flat floor). |
| `intel_xpu_resource_leak_bug_report.md` | UR-handle leak → `OUT_OF_RESOURCES` **crash** ~iter 70. Crash, not hang. |

## Open technical gap (crux of the ticket's Ask #1)

The updated ticket rightly decouples *the deadlock mechanism* (an FSDP collective desync that can't
tolerate a transiently-slow rank) from *why a rank lags*. But the unresolved point is that a purely
**transient** slowdown on a **matched** collective cannot produce a **permanent** hang — the peers
wait, the straggler arrives, the collective drains. Something must convert the delay into an ordering
mismatch, and the captured stacks show two DIFFERENT collectives on two DIFFERENT PGs:
- 114 ranks in cross-node `all_reduce` on the **replicate PG** (backward `_reduce_grad`);
- 10 ranks in within-node `all_gather` on the **shard PG** (forward `_pre_forward_unshard`).

These two readings imply **opposite answers to Ask #1**, and one cheap diagnostic distinguishes them
— **which nodes are the 10/24 forward-unshard ranks on** (the per-incident nodefiles are already
attached to the ticket):
- concentrated on 1–2 nodes → within-node shard-PG **collective-order divergence** → no CCL
  timeout+retry can fix a mismatch; the fix is FSDP-schedule-side.
- scattered ~1/node → cross-PG **straggler cascade** (a node's stalled shard-PG all_gather emits no
  rank into the replicate all_reduce → other 15 nodes hang) → a replicate-`all_reduce` timeout is a
  plausible mitigation.

Recommend adding that rank-locality breakdown (`awk` over the attached nodefile) before filing — it
turns Ask #1 from "is there a setting?" into a question ALCF can answer decisively. The phrase
"cannot tolerate a transiently-slow rank" presumes the same-collective reading; back it with the
node-locality data.

## Cross-reference

Full ticket + per-incident nodefiles + stack dumps:
`/flare/ModCon/ngetty/vjepa2/docs/ALCF_TICKET_fsdp_allreduce_deadlock.md` and
`.../checkpoints/surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak/hang_diag/`.
Same silent-hang *class* other Aurora large-scale jobs report (AGPT/torchtitan 256N
"blind-rotate a node"); this one has a captured traceback.
