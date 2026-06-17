# AGPT-2B 8N HSDP — step-based resume + growth-past-150 investigation

**Date:** 2026-06-17
**Status:** Resume SHIPPED + validated. Growth-past-150 investigated: marginal at this
envelope (capability ceiling, not a step-count wall).

## Part 1 — Standard step-based resume (SHIPPED)

GRPO step-based runs (epochs=1, num_steps=N) are now resumable to continue past their step
budget with full continuity. Validated end-to-end on 8N HSDP: fresh 150-step run (job
8545903, rc=0) → resume to 300 (job 8545985 at lr=5e-6, then 8546049 at lr=1e-5), both
rc=0, first resumed step=151, optimizer momentum continuous across the boundary (grad_norm
5.6→4.9, no fresh-optimizer spike), recipe_state re-resumable at every checkpoint.

Four resume bugs fixed (each only surfaced by running the full save→resume cycle; CPU tests
passed throughout):
1. `intermediate_checkpoint` now also true when `save_every_n_steps` is set → step-based
   runs write recipe_state (opt-in; runs without it stay weights-only / byte-identical).
2. `training.STEPS_KEY` persisted + restored; setup() derives `_steps_run` on resume instead
   of resetting global_step → loop continues from the saved step.
3. Policy/ref checkpointer split (`load_checkpoint(store_as=...)`, ref loads `resume=False`)
   → checkpoints land in `${output_dir}/epoch_N` not `/ref/`; ref no longer asserts on a
   missing recipe_state.
4. FSDP1-native optimizer state (`FSDP.optim_state_dict` / `optim_state_dict_to_load`) on the
   FSDP1 path — the DCP save / manual-DTensor load were asymmetric for `use_orig_params`,
   crashing `optimizer.step()` with a shape mismatch. Plus: don't increment `_epochs_run` /
   re-save when stopping on the step budget (was poisoning `epochs_run=1` → empty resume
   loop); clamp `_epochs_run` on resume if advanced past total_epochs but steps remain.

Bonus capability (validated): **resume at a different lr / constant lr.** After the optimizer
load restores the checkpoint's saved lr, the config lr is re-applied to param_groups; and
`lr_scheduler=null` is tolerated (defensive guard) to disable cosine and run constant-lr.

**Usage:** `RESUME_FROM=<output_dir>/epoch_0` on the launcher (config sets
`save_every_n_steps=50`). To change lr on resume:
`EXTRA_OVERRIDES="optimizer.lr=1e-5 lr_scheduler=null"`.

CPU pin-down: `tests/torchtune/dev/rl/test_checkpoint_resume_state.py` (14 tests). Recipe:
`recipes/dev/grpo_full_finetune_distributed_xpu.py`.

## Part 2 — Does AGPT-2B grow past 150 steps at 8N? (marginal)

Late-window reward/success (single seed each; SFT-variance band historically ~4-7% rel):

| run | window | reward | success |
|---|---|---|---|
| 8N lr=5e-6 (job 8545903) | 100-150 | 0.188 | 0.083 |
| 8N lr=5e-6 resume (8545985) | 250-300 | 0.179 | 0.069 |
| **8N lr=1e-5 resume (8546049)** | **250-300** | **0.195** | **0.091** |
| 8N lr=1e-5 resume | 275-300 | 0.184 | 0.079 |

- lr=5e-6 **plateaus** by ~150 and the 150→300 continuation drifts flat-to-down.
- lr=1e-5 (constant, the lever that broke the 2N plateau) is **slightly better** than both
  (~+10% rel over the plateau) but **within single-seed noise** — NOT the decisive breakout
  the 2N run showed (+2.4pp success over 150→225). KL stayed bounded (0.15-0.29, no blow-up),
  so this is genuine convergence, not instability.

### Interpretation
The 8N run sees 14 distinct prompts/step (vs 2N's 1), so 150 8N-steps ≈ 2100 2N-prompt
exposures — it is effectively already deep in training by step 150 and converges faster +
flatter than 2N. The flat tail at the same ~8-9% success that capped 2N suggests AGPT-2B has
hit its **GSM8K capability ceiling** for this SFT-init + reward setup. The lever to push past
~9% is therefore a capability change (better SFT init, denser/harder reward, more capable
base) — NOT more GRPO steps. Resume now makes long continuations cheap to try, but the data
says steps alone won't move it much here.

## Combined chapter result
- 8N HSDP scale-up (prior report): ~10× distinct-prompt throughput AND +26% reward / +53%
  success vs 2N at equal steps.
- Step-based resume: shipped + validated (continue indefinitely, optionally at a new lr).
- Growth past 150: capped by model/task capability at this envelope, not by steps.
