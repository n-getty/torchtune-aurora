# Repo feedback compute smoke validation — 2026-04-30

Validates the repo_feedback.txt fix batch (A/B/C/D) on real Aurora compute.
Ran on held debug-scaling node `x4416c0s0b0n0` (job 8461906).

Scripts: `experiments/repo_feedback/run_smoke{1,2,3}_*.sh`
Logs: `experiments/repo_feedback/smoke{1,2,3}_*.log`

## Results

| Smoke | Mode                                | Steps | Exit | Time     | Notes |
|-------|-------------------------------------|-------|------|----------|-------|
| 1     | dense colocate, sync, Qwen2.5-3B    | 5/5   | 0    | 207s wall, ~9s/step | ratios=1.0000, no errors |
| 2     | vLLM server (TP=2) + wsync (shm)    | 5/5   | 0    | ~2:07 train, 4 wsync events | sync_ids 1-4, ~2.3s/sync warm |
| 3     | resume from smoke 2 ckpt + 3 steps  | 8/8   | 0    | ~3 steps post-resume | A1 NameError fix VALIDATED |

## What this proves

- **A1**: Real-compute regression test for the `OPT_KEY`/`DATALOADER_KEY`
  cleanup ordering bug. Resume completed without `NameError` and produced
  3 additional optimizer steps. Confirms `self._opt_state_dict` /
  `self._dataloader_state_dict` capture before `checkpoint_dict.clear()`
  works under both policy AND ref checkpointer load.
- **C2**: Weight version bumps showed `sync_id=1..4` in logs — the
  `_pending_sync_id` precision tracking is firing exactly once per dispatched
  sync, no double-counting.
- **vllm_backend `_xpu_device_index` cross-module bug**: surfaced and fixed
  during smoke 1 (was a NameError on every rank at engine init). Replaced the
  cross-module global lookup with local `ZE_AFFINITY_MASK`/`LOCAL_RANK`
  derivation in `torchtune/dev/rl/vllm_backend.py:39-44`.

## Recipe quirk (not a bug, but documented for future me)

`load_checkpoint(cfg.ref_checkpointer)` at line 883 reassigns `self._checkpointer`
to the ref checkpointer instance. Result: `save_checkpoint()` writes to
`${output_dir}/ref/epoch_N`, not `${output_dir}/epoch_N`. Smoke 3 had to
override `checkpointer.output_dir=${output_dir}/ref` so the resume's
`get_recipe_checkpoint_path` would find the saved `recipe_state.pt`.

This is preexisting behavior and out of scope for the repo_feedback work,
but worth knowing if you ever debug "where did my checkpoint save?".

## Self-grep false positive in run_smoke3_resume.sh (fixed)

The script's `grep -q "NameError" "${LOG}"` matched its own diagnostic
header line ("NameError check (the bug A1 fixed):") that had been tee'd
into the log, producing a spurious FAIL printout while the actual run was
clean. Fixed to match `^NameError:|raise NameError` only.
