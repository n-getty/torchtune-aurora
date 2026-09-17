# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test for the BioReason SFT LR-schedule resume bug.

`recipes/dev/sft_bioreason_distributed_xpu.py`'s `setup()` used to restore
`self.global_step` from `_resume_blob` AFTER calling `super().setup()`. But the
parent (`recipes/dev/full_finetune_distributed_xpu.py:641-644`) builds the LR
scheduler with `last_epoch=self.global_step - 1` *inside* that `super().setup()`
call — so at build time `global_step` was still 0, `last_epoch` was -1, and every
resume segment silently restarted the cosine schedule from warmup. The fix moves
the restore into `_setup_model` (called by the parent's own `setup()`, before the
scheduler is built), so `_resume_blob["steps"]` is available in time.

Separately, `num_training_steps` for the cosine schedule defaulted to
`total_epochs * steps_per_epoch`. Some launchers set `epochs` far above the
number of steps a run will actually execute (deliberately, to avoid exiting
early on a scaled-up allocation), which makes the cosine horizon far longer
than the real run — so even with the resume bug fixed, the schedule barely
anneals. `lr_scheduler_num_training_steps` is now a separate, explicit config
key for the real step budget.

This test drives the actual `get_cosine_schedule_with_warmup` (no XPU, no
recipe object, no mocks of scheduler math) against a toy optimizer to pin down
both fixes.
"""
import torch

from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup


def _make_optimizer(lr=1e-4):
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


def _build_scheduler(*, num_warmup_steps, num_training_steps, last_epoch):
    optimizer = _make_optimizer()
    if last_epoch != -1:
        # Mirrors a real resume: the optimizer state dict (loaded before the
        # scheduler is built) already carries "initial_lr" in each param
        # group from the prior run's scheduler construction. LRScheduler
        # requires this whenever last_epoch >= 0.
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch,
    )
    return optimizer, scheduler


def _run_to_step(*, num_warmup_steps, num_training_steps, start_step, end_step):
    """Simulate stepping a scheduler from `start_step` to `end_step` (exclusive of
    the initial state, inclusive of the final .step() call), mirroring how the
    recipe's train loop calls lr_scheduler.step() once per optimizer step after
    building it with last_epoch=start_step - 1."""
    optimizer, scheduler = _build_scheduler(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=start_step - 1,
    )
    for _ in range(end_step - start_step):
        optimizer.step()
        scheduler.step()
    return optimizer.param_groups[0]["lr"]


def test_resume_lr_matches_uninterrupted_run():
    """A1: LR at step N after a simulated resume (global_step correctly restored
    before scheduler build) must equal LR at step N in an uninterrupted run."""
    num_warmup_steps = 50
    num_training_steps = 2000
    resume_step = 800
    target_step = 1200

    uninterrupted_lr = _run_to_step(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        start_step=0,
        end_step=target_step,
    )
    resumed_lr = _run_to_step(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        start_step=resume_step,
        end_step=target_step,
    )
    assert resumed_lr == uninterrupted_lr


def test_unfixed_resume_restarts_at_warmup():
    """Demonstrates the bug this regression test guards against: building the
    scheduler with last_epoch=-1 (global_step not yet restored) at a resume
    boundary gives the FIRST warmup-step LR, not the true schedule position —
    this is what shipped before A1."""
    num_warmup_steps = 50
    num_training_steps = 2000
    resume_step = 800

    _, buggy_scheduler = _build_scheduler(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=-1,  # bug: global_step restored too late, so last_epoch defaults here
    )
    buggy_lr_at_resume = buggy_scheduler.get_last_lr()[0]

    _, fixed_scheduler = _build_scheduler(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=resume_step - 1,  # fix: global_step correctly restored first
    )
    fixed_lr_at_resume = fixed_scheduler.get_last_lr()[0]

    # last_epoch=-1 evaluates the warmup lambda at current_step=0 -> LR literally
    # collapses to 0.0 at the resume boundary, not merely "low".
    assert buggy_lr_at_resume == 0.0
    assert fixed_lr_at_resume > buggy_lr_at_resume


def test_horizon_set_to_real_budget_anneals_near_eta_min():
    """A2: with num_training_steps set to the actual step budget (not an
    allocation-filling `epochs * steps_per_epoch` far beyond it), LR at the
    final step must be near eta_min (~0). This is what would have caught the
    near-null version of the v6/v7 experiment: fixing only the resume bug but
    leaving the horizon at epochs=50's inflated step count keeps LR at 86-97%
    of peak for the whole run, per the plan's table."""
    num_warmup_steps = 50
    num_training_steps = 1525  # the real step budget, not epochs * steps_per_epoch
    peak_lr = _make_optimizer().param_groups[0]["lr"]

    final_lr = _run_to_step(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        start_step=0,
        end_step=num_training_steps,
    )
    assert final_lr < 0.01 * peak_lr


def test_horizon_left_at_inflated_epoch_budget_barely_anneals():
    """Negative control: mirrors the plan's table row (v6 8N step 1500 with only
    A1 fixed, horizon left at epochs=50's 16150-step schedule) — LR stays close
    to peak, which is the near-null-test failure mode A2 exists to prevent."""
    num_warmup_steps = 50
    num_training_steps = 16150  # epochs=50 * steps_per_epoch, NOT the real budget
    real_step_reached = 1500
    peak_lr = _make_optimizer().param_groups[0]["lr"]

    lr_at_1500 = _run_to_step(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        start_step=0,
        end_step=real_step_reached,
    )
    assert lr_at_1500 > 0.9 * peak_lr


def test_resume_after_real_checkpoint_roundtrip_does_not_double_decay():
    """Regression test for a bug this session's A2 fix newly exposed on real
    hardware (job 8799523, 2026-09-03): a resume across a FRESH process (not
    just a fresh scheduler in the same process) reconstructs the optimizer from
    scratch and loads its state dict via
    torchtune.training.load_from_full_optimizer_state_dict. That round-trip
    restores "lr" (now DECAYED, e.g. 9.747e-05 partway through a cosine
    schedule) into each param_group, but does NOT guarantee "initial_lr"
    survives — so building the scheduler with last_epoch >= 0 crashes with
    KeyError on every rank (confirmed: all 192 ranks in job 8799523).

    The naive fix — group.setdefault("initial_lr", group["lr"]) using the
    CURRENT (decayed) lr, exactly as this file's own _build_scheduler helper
    above does — is WRONG here: it would make the scheduler treat the already
    -decayed value as the new peak and decay a SECOND time from there. The
    real fix (recipes/dev/full_finetune_distributed_xpu.py's
    _setup_lr_scheduler) must recover the true peak LR from the config
    (self._optimizer_peak_lr, captured before optimizer construction), not
    from the live optimizer state.

    This test drives that exact sequence with real torch scheduler math: run a
    schedule partway to a decayed LR, simulate a fresh-process checkpoint
    round-trip (new optimizer, "initial_lr" NOT present, "lr" = the decayed
    value), then verify continuing with the CORRECT initial_lr (peak, from
    "config") reproduces the uninterrupted schedule, while the naive
    (current-lr) fix would not.
    """
    num_warmup_steps = 50
    num_training_steps = 2000
    resume_step = 800
    target_step = 1200
    peak_lr = 1e-4

    # Uninterrupted reference: same schedule, run straight through.
    uninterrupted_lr = _run_to_step(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        start_step=0,
        end_step=target_step,
    )

    # Simulate the real failure: run to resume_step, capture the DECAYED lr
    # (this is what the checkpoint would contain), and confirm it is indeed
    # below peak (otherwise this test wouldn't be exercising the real bug).
    optimizer, scheduler = _build_scheduler(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=-1,
    )
    for _ in range(resume_step):
        optimizer.step()
        scheduler.step()
    decayed_lr_at_checkpoint = optimizer.param_groups[0]["lr"]
    assert decayed_lr_at_checkpoint < peak_lr, (
        "test setup invalid: checkpoint lr must be decayed below peak for this "
        "test to exercise the double-decay bug"
    )

    # Simulate a FRESH process's optimizer reconstruction + state-dict load:
    # a brand new optimizer, with "lr" set to the checkpoint's decayed value
    # and NO "initial_lr" key (exactly what
    # load_from_full_optimizer_state_dict's round-trip produces).
    fresh_param = torch.nn.Parameter(torch.zeros(1))
    fresh_optimizer = torch.optim.SGD([fresh_param], lr=decayed_lr_at_checkpoint)
    assert "initial_lr" not in fresh_optimizer.param_groups[0]

    # THE FIX: initial_lr must come from the config's peak lr, not the live
    # (decayed) group["lr"].
    for group in fresh_optimizer.param_groups:
        group.setdefault("initial_lr", peak_lr)
    fixed_scheduler = get_cosine_schedule_with_warmup(
        fresh_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=resume_step - 1,
    )
    for _ in range(target_step - resume_step):
        fresh_optimizer.step()
        fixed_scheduler.step()
    fixed_resumed_lr = fresh_optimizer.param_groups[0]["lr"]
    assert fixed_resumed_lr == uninterrupted_lr

    # THE BUG (naive fix): initial_lr = current decayed lr would double-decay.
    fresh_param2 = torch.nn.Parameter(torch.zeros(1))
    buggy_optimizer = torch.optim.SGD([fresh_param2], lr=decayed_lr_at_checkpoint)
    for group in buggy_optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])  # BUG: uses decayed lr
    buggy_scheduler = get_cosine_schedule_with_warmup(
        buggy_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=resume_step - 1,
    )
    for _ in range(target_step - resume_step):
        buggy_optimizer.step()
        buggy_scheduler.step()
    buggy_resumed_lr = buggy_optimizer.param_groups[0]["lr"]
    assert buggy_resumed_lr != uninterrupted_lr
    assert buggy_resumed_lr < fixed_resumed_lr
