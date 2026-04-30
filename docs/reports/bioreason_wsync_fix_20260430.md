# BioReason 2-node Phase 2: weight sync correctness fix + G-bisect

**Date**: 2026-04-30
**Hold**: 8460176 (debug-scaling, 2 nodes, 1h)
**Recipe**: `recipes/dev/grpo_bioreason_distributed_xpu.py`
**Config**: `recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml`

## Summary

Two findings, in order of importance:

1. **Correctness bug in `train()` override (FIXED, validated)**: the BioReason recipe's
   `train()` method never invoked `_sync_weights_to_vllm`. vLLM rollouts in earlier
   "stable" runs (47, 50, 51) were generated from SFT-initial weights for the entire
   run — training optimized the policy normally, but rollouts never saw any update.
   The fix is a 10-line block after `optimizer.step()` that mirrors the base recipe.
   Validated 5/5 clean on G=4 fbs=4 max_gen=1024.

2. **G-bisect characterizes G>4 blocker**: with parallel HTTP fan-out enabled
   (run 51), only G=4 reaches steady state. The crash step is inversely correlated
   with G — G=8 dies at step 13, G=12 at step 3, G=16 at step 2 — and the signature
   is the documented `banned:1` PDE Segfault from CCL/L0 IPC-handle accumulation
   (same root-cause class as `memory/bugs/project_ccl_ipc_handle_cache.md`).
   **Production envelope locks at G=4** until a permanent IPC-handle leak fix lands.

---

## Bug 1: missing `_sync_weights_to_vllm` call in `train()` override

### Symptom

- All earlier "successful" 2-node Phase 2 runs (47/50/51) showed flat KL (0.002-0.006)
  and `loss=0.0000` essentially throughout.
- The `_metric_logger` shows the policy gradient is non-zero, which suggested training
  was working; the flat KL was attributed to "max_gen too short / reward sparse."
- No log line about weight sync ever appeared in the run-50 / run-51 transcripts.

### Root cause

`recipes/dev/grpo_bioreason_distributed_xpu.py:1288` overrides the base `train()`
loop. The override correctly handles the multimodal-specific generate/grpo path
(prompt_embeds, vLLM HTTP fan-out, BioReason-specific rewards) but **omits the
post-optimizer weight-sync block** that the base recipe runs at lines 3380-3392
of `grpo_full_finetune_distributed_xpu.py`:

```python
# base recipe — present
if self._vllm_mode == "colocate":
    self._sync_colocated_weights()
elif self._vllm_mode == "dedicated_rank" and not self._is_vllm_rank:
    self._sync_dedicated_vllm_weights()
elif self._vllm_mode == "server" and self._vllm_weight_sync:
    if self._steps_run % self._vllm_weight_sync_interval == 0:
        self._wait_for_sync_complete()
        self._sync_weights_to_vllm()
```

The BioReason override jumped from `optimizer.step() / zero_grad()` straight to
`cleanup_after_step()`, so this block never ran in `server` mode (the production
2-node config). The vLLM server kept the weights it loaded at startup for the
duration of training. Generations were forever from a fixed reference distribution.

### Why it was hidden

- The metric_logger reports loss/KL/policy_loss numbers that are perfectly
  computable on stale rollouts — there's no "rollout staleness" telemetry.
- Step time and memory looked clean (run 51: 58s/step, FLAT), which read as
  success against the "make it not crash" success criterion of Phase 2 prototyping.
- The `_run_vllm_generation_server()` loop comment at `weight_sync.py:539` says
  "Receive weight update from rank 0 via wsync_pg," which is true for
  `dedicated_rank` mode. In `server` mode that line is a no-op (HTTP path,
  no PG); the actual trigger is the omitted `_sync_weights_to_vllm()` call
  on the *training* side.

### Fix

`recipes/dev/grpo_bioreason_distributed_xpu.py` adds the same dispatch block
after `self._steps_run += 1`:

```python
if getattr(self, "_vllm_mode", None) == "dedicated_rank" and not self._is_vllm_rank:
    self._sync_dedicated_vllm_weights()
elif (
    getattr(self, "_vllm_mode", None) == "server"
    and getattr(self, "_vllm_weight_sync", False)
):
    if self._steps_run % self._vllm_weight_sync_interval == 0:
        self._wait_for_sync_complete()
        self._sync_weights_to_vllm()
```

`_sync_weights_to_vllm` (in `torchtune/dev/rl/weight_sync.py:747`) already has
BioReason-specific logic: filters to `backbone.*` only (399 params for Qwen3-4B),
strips FSDP/AC wrappers. `_sync_done_event` is initialized in
`vllm_backend.py:542` for server mode, so `_wait_for_sync_complete()` is safe.

### Validation (hold 8460176, G=4 fbs=4 max_gen=1024 NSTEPS=5)

| Step | total | gen | grpo | gather | save | mem reserved |
|------|-------|-----|------|--------|------|--------------|
| 1 | 44.8s | 29.1 | 15.5 | 5.2s | 6.4s @ 1.3 GB/s | 45.33 GiB |
| 2 | 58.3s | 44.1 | 13.8 | 5.1s | 6.9s @ 1.2 GB/s | 52.87 GiB |
| 3 | 51.3s | 37.6 | 13.4 | 5.1s | 7.5s @ 1.1 GB/s | 52.87 GiB |
| 4 | 50.6s | 36.7 | 13.5 | 5.1s | 8.0s @ 1.0 GB/s | 52.87 GiB |
| 5 | 52.3s | 37.3 | 14.5 | 5.1s | – | – |

- Weight version bumped 1→5 (one sync per step; `interval=1` honored).
- 399 backbone params gathered every step (correct: Qwen3-4B backbone only).
- Memory FLAT at 52.87 GiB across steps 2-5 — no leak introduced by the fix.
- Per-step ~50s vs 43s run-47 baseline (+7s = pure gather window; save+POST
  overlapped with next generation).
- `BioReason GRPO training complete after 5 steps.` (clean exit).

### Open observation (separate bug, not wsync)

`loss=0.0000` and KL 0.002-0.003 stayed flat across all 5 steps even with sync
working. Two non-exclusive hypotheses:

1. `max_gen_tokens=1024` still truncates before the model emits the
   `<gene_recall>...</gene_recall>` block in many trajectories → reward signal
   stays zero → all advantages in a group equal → policy gradient zero. This
   matches the prior `bugs/project_bioreason_replicated_data_bug.md` finding
   that 512 tokens was clearly too short.
2. Reward is sparse enough that the GRPO advantage normalization (subtract
   group mean, divide by group std) collapses to zero whenever all G=4 sequences
   in a group earn the same reward.

Disambiguation needs reward instrumentation (mean/var per group), not more sync work.

### Application rule for future model recipes

Any recipe that subclasses the general GRPO recipe and overrides `train()`
**must replicate the post-optimizer weight-sync dispatch block verbatim**.
Don't rely on the base loop being reused. Recorded as project-level rule in
`memory/bugs/project_bioreason_train_missing_wsync.md`.

---

## Bug 2: G>4 banned:1 PDE Segfault (G-bisect)

### Setup

After the run-51 parallel POST fan-out optimization landed (2026-04-29; reduced
58s/step at G=4), we tried to push G higher to amortize generation cost across
more rollouts. All tests on fresh holds with `TORCHTUNE_USE_CHUNKED_LOSS=1` and
`fbs` matched to `G` (single chunk per micro-batch).

### Bisect results

| G | Steps clean | Crash signature |
|---|-------------|-----------------|
| 4 | 15+ | none, mem FLAT 52.84 GiB |
| 8 | 12 | step 13: `grpo` spike 20s→52.6s, mem DROP 62.28→51.44, banned:1 ranks 2,3,8 |
| 12 | 2 | step 3: `grpo` spike 30s→84s, then banned:1 ranks 5,10 |
| 16 | 1 | step 2: `grpo`=78s at step 1, then banned:1 ranks 0,1,3,5,9,10 |

The "slow step then crash" pattern (sudden +30-60s jump in `grpo` time, mem
contraction, then PDE) is the documented L0/CCL eviction-thrash signal that
precedes a `banned:1` write fault.

### Falsified hypotheses

- **HTTP fan-out concurrency**: added `TORCHTUNE_VLLM_FANOUT_MAX=4` env var
  capping the ThreadPoolExecutor to 4 simultaneous POSTs (matches G=4 worker
  count). G=16 with cap=4 crashed identically at step 2. Concurrency of HTTP
  POSTs is not the variable.
- **Chunk count**: G=16 fbs=4 (4 chunks) crashed identically to G=16 fbs=16
  (1 chunk). Chunk count is not the variable.
- **Walltime / IPC handle warmup**: G=4 ran 15+ steps clean over 11 minutes
  on the same hold class — wall-clock duration alone doesn't trigger it.

### Conclusion

What does scale with G is **per-step CCL IPC-handle creation traffic** — every
forward chunk allocates new tensors that get registered as IPC handles for the
next FSDP AllGather/ReduceScatter. At higher G, more handles per step land in
the L0 driver's IPC handle cache. The crash step is inversely correlated with G,
which is the IPC-handle-budget signature. Same root-cause class as
`memory/bugs/project_ccl_ipc_handle_cache.md` (the late-step `banned:1` story
on 32B dense).

### Operational rule

- **Production envelope: G=4 fbs=4 max_gen=1024.** Only stable BioReason 2-node
  config until a permanent IPC-handle leak fix targets per-step G-scaling
  handle creation in `_generate_with_vllm_server_embeds`.
- **Hold side-effect**: any `banned:1` crash makes that hold's L0 device state
  unrecoverable. `torch.xpu.device_count()` returns 0 even after `pkill -9`
  and `clean_tiles.sh --kill`. Fresh PBS allocation required. Always queue a
  backup hold before risky G-sweeps. Recorded in
  `memory/feedback_banned1_destroys_xpu.md`.

---

## Files changed

- `recipes/dev/grpo_bioreason_distributed_xpu.py` — added wsync dispatch block
  in `train()` override.
- `experiments/bioreason/run_bioreason_2node_server.sh` — added
  `TORCHTUNE_VLLM_FANOUT_MAX` passthrough (used during G-bisect; harmless when
  unset).

## Memory updates

- `memory/bugs/project_bioreason_train_missing_wsync.md` — new bug entry.
- `memory/feedback_banned1_destroys_xpu.md` — new feedback entry.
- `memory/project_bioreason_overnight_2026_04_30.md` — full G-bisect log.
- `memory/MEMORY.md` — index entries for the above.

## Next steps (suggested, not done)

1. **Reward-side flat-KL diagnosis** at G=4 with the wsync fix in place: instrument
   group reward mean/var and trajectory `response_length` distribution; confirm
   whether all groups have zero variance or whether the issue is truncation.
2. **Decide on long-horizon validation**: with the wsync fix, KL behavior should
   change relative to runs 47/50/51. A 4h capacity run at G=4 would either show
   meaningful KL drift (real RL) or confirm the reward-side blocker independent
   of wsync.
3. **Defer G>4 until IPC-handle leak fix lands**. The signal is now well-enough
   characterized that the next investigation should be on the CCL/L0 side
   (handle creation per-chunk in `_generate_with_vllm_server_embeds` and
   the FSDP collective path), not on more BioReason-side mitigations.
