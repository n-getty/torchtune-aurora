# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test for the checkpoint-resume cleanup bug.

Before the fix, `setup()` deleted `checkpoint_dict` after the policy model load
but later tried to read `checkpoint_dict[training.OPT_KEY]` and
`[training.DATALOADER_KEY]` from inside `_setup_optimizer` / `_setup_data` calls
that are gated by `self._resume_from_checkpoint`. That raised NameError on
resume.

This test exercises the lifecycle pattern (capture-before-clear → consume-from-self)
without needing XPU or a real recipe object.
"""
from unittest.mock import MagicMock

from torchtune import training


class _FakeRecipe:
    """Mirror the lifecycle used by GRPOFullFinetuneDistributedXPU.setup()."""

    def __init__(self, resume: bool):
        self._resume_from_checkpoint = resume
        self._opt_state_dict = None
        self._dataloader_state_dict = None

    def setup(self, checkpoint_dict):
        # Mirrors recipes/dev/grpo_full_finetune_distributed_xpu.py:
        # capture state dicts BEFORE the cleanup that releases checkpoint_dict.
        self._opt_state_dict = (
            checkpoint_dict.get(training.OPT_KEY)
            if self._resume_from_checkpoint
            else None
        )
        self._dataloader_state_dict = (
            checkpoint_dict.get(training.DATALOADER_KEY)
            if self._resume_from_checkpoint
            else None
        )
        # Simulate the cleanup that previously deleted checkpoint_dict.
        if training.MODEL_KEY in checkpoint_dict:
            checkpoint_dict[training.MODEL_KEY] = None
        checkpoint_dict.clear()
        del checkpoint_dict

        # Consumers run AFTER the cleanup — they must not reach back into the
        # (now-deleted) checkpoint_dict; they read from self.
        opt_received = self._setup_optimizer(opt_state_dict=self._opt_state_dict)
        self._opt_state_dict = None
        dl_received = self._setup_data(dataloader_state_dict=self._dataloader_state_dict)
        self._dataloader_state_dict = None
        return opt_received, dl_received

    def _setup_optimizer(self, opt_state_dict):
        return opt_state_dict

    def _setup_data(self, dataloader_state_dict):
        return dataloader_state_dict


def _build_ckpt():
    return {
        training.MODEL_KEY: {"layer.weight": "stub-tensor"},
        training.OPT_KEY: MagicMock(name="opt_state"),
        training.DATALOADER_KEY: MagicMock(name="dl_state"),
    }


def test_resume_consumers_receive_state_after_cleanup():
    ckpt = _build_ckpt()
    expected_opt = ckpt[training.OPT_KEY]
    expected_dl = ckpt[training.DATALOADER_KEY]

    recipe = _FakeRecipe(resume=True)
    opt, dl = recipe.setup(ckpt)

    assert opt is expected_opt, "opt_state_dict must survive cleanup on resume"
    assert dl is expected_dl, "dataloader_state_dict must survive cleanup on resume"


def test_no_resume_passes_none_to_consumers():
    ckpt = _build_ckpt()
    recipe = _FakeRecipe(resume=False)
    opt, dl = recipe.setup(ckpt)
    assert opt is None
    assert dl is None


def test_state_released_after_consumption():
    """References should be cleared once consumed so they don't pin memory."""
    ckpt = _build_ckpt()
    recipe = _FakeRecipe(resume=True)
    recipe.setup(ckpt)
    assert recipe._opt_state_dict is None
    assert recipe._dataloader_state_dict is None


# ---------------------------------------------------------------------------
# Step-based resume (STEPS_KEY persist/restore + intermediate_checkpoint gate)
# Mirrors the save_checkpoint gate, _update_recipe_state restore, and the
# setup() step-counter derivation in
# recipes/dev/grpo_full_finetune_distributed_xpu.py. Pure logic — no XPU/recipe.
# ---------------------------------------------------------------------------


class _FakeCheckpointLifecycle:
    """Mirror the save-gate / restore / step-derivation logic."""

    def __init__(self, *, epochs, num_steps, save_every_n_steps, resume):
        self.total_epochs = epochs
        self._total_steps = num_steps
        self._save_every_n_steps = save_every_n_steps
        self._resume_from_checkpoint = resume
        self.global_step = 0
        self._steps_run = 0
        self._epochs_run = 0

    # --- save side (mirrors save_checkpoint ~2378 + dict block) -------------
    def intermediate_checkpoint(self, epoch):
        return (epoch + 1 < self.total_epochs) or (
            self._save_every_n_steps is not None
        )

    def build_recipe_state(self, epoch):
        if not self.intermediate_checkpoint(epoch):
            return {training.MODEL_KEY: "weights"}  # weights-only
        return {
            training.MODEL_KEY: "weights",
            training.OPT_KEY: "opt",
            training.EPOCHS_KEY: self._epochs_run,
            training.STEPS_KEY: self.global_step,
            training.DATALOADER_KEY: "dl",
        }

    # --- restore side (mirrors _update_recipe_state ~866) ------------------
    def update_recipe_state(self, ckpt):
        self._epochs_run = ckpt[training.EPOCHS_KEY]
        self.global_step = ckpt.get(training.STEPS_KEY, 0)

    # --- setup() step-counter derivation (~1149) --------------------------
    def derive_step_counters(self, steps_per_epoch):
        self._steps_per_epoch = steps_per_epoch
        if self._resume_from_checkpoint:
            # Clamp _epochs_run if it advanced past total_epochs but steps remain
            # (prior run stopped on step budget then incremented the counter),
            # else the epoch loop would be empty and resume runs 0 steps.
            if (
                self._epochs_run >= self.total_epochs
                and self.global_step < self._total_steps
            ):
                self._epochs_run = self.total_epochs - 1
            self._steps_run = (
                self.global_step - self._epochs_run * self._steps_per_epoch
            )
        else:
            self.global_step = self._epochs_run * self._steps_per_epoch

    def epoch_loop_runs(self):
        # Mirrors `for curr_epoch in range(self._epochs_run, self.total_epochs)`.
        return self._epochs_run < self.total_epochs


def test_intermediate_checkpoint_true_for_step_based_run():
    # epochs=1, num_steps=150, save_every_n_steps set → resumable even on epoch 0.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=150, save_every_n_steps=50, resume=False
    )
    assert r.intermediate_checkpoint(epoch=0) is True


def test_no_recipe_state_when_save_every_n_steps_unset():
    # Single-shot config (no save_every_n_steps) keeps weights-only final save.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=150, save_every_n_steps=None, resume=False
    )
    assert r.intermediate_checkpoint(epoch=0) is False
    state = r.build_recipe_state(epoch=0)
    assert training.STEPS_KEY not in state
    assert training.OPT_KEY not in state


def test_steps_key_saved_and_round_trips():
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=300, save_every_n_steps=50, resume=False
    )
    r.global_step = 150
    state = r.build_recipe_state(epoch=0)
    assert state[training.STEPS_KEY] == 150

    # Resume run consumes it.
    r2 = _FakeCheckpointLifecycle(
        epochs=1, num_steps=300, save_every_n_steps=50, resume=True
    )
    r2.update_recipe_state(state)
    assert r2.global_step == 150


def test_missing_steps_key_tolerated():
    # Older recipe_state without STEPS_KEY → global_step falls back to 0.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=300, save_every_n_steps=50, resume=True
    )
    r.update_recipe_state(
        {training.EPOCHS_KEY: 0, training.OPT_KEY: "opt"}
    )
    assert r.global_step == 0


def test_resume_derives_steps_run_continues_not_restarts():
    # The crux: resume at step 150 (epochs_run=0) → _steps_run=150 so the loop
    # continues to num_steps=300 rather than re-running from 0.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=300, save_every_n_steps=50, resume=True
    )
    r.update_recipe_state(
        {training.EPOCHS_KEY: 0, training.STEPS_KEY: 150, training.OPT_KEY: "opt"}
    )
    r.derive_step_counters(steps_per_epoch=1000)
    assert r._steps_run == 150  # continues, does not restart


def test_fresh_run_step_counter_identical_to_original():
    # Non-resume path must be byte-identical to the original behavior.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=300, save_every_n_steps=50, resume=False
    )
    r._epochs_run = 0
    r.derive_step_counters(steps_per_epoch=1000)
    assert r.global_step == 0  # original: _epochs_run * steps_per_epoch


def test_resume_clamps_advanced_epoch_so_loop_runs():
    # The job-1 bug: epochs_run got saved as 1 (== total_epochs) after stopping
    # on the step budget. On resume with num_steps bumped to 300, the epoch loop
    # would be range(1,1) → empty → 0 steps. The clamp must let it run.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=300, save_every_n_steps=50, resume=True
    )
    r.update_recipe_state(
        {training.EPOCHS_KEY: 1, training.STEPS_KEY: 150, training.OPT_KEY: "opt"}
    )
    assert r.epoch_loop_runs() is False  # before clamp: range(1,1) empty
    r.derive_step_counters(steps_per_epoch=1000)
    assert r._epochs_run == 0  # clamped
    assert r.epoch_loop_runs() is True  # now range(0,1) runs
    assert r._steps_run == 150  # continues from 150


def test_resume_no_clamp_when_budget_already_met():
    # If the restored global_step already >= num_steps, do NOT clamp — there is
    # genuinely nothing left to do; the (empty) loop correctly does nothing.
    r = _FakeCheckpointLifecycle(
        epochs=1, num_steps=150, save_every_n_steps=50, resume=True
    )
    r.update_recipe_state(
        {training.EPOCHS_KEY: 1, training.STEPS_KEY: 150, training.OPT_KEY: "opt"}
    )
    r.derive_step_counters(steps_per_epoch=1000)
    assert r._epochs_run == 1  # NOT clamped (no steps owed)
    assert r.epoch_loop_runs() is False


# ---------------------------------------------------------------------------
# Resume-at-a-different-lr: (1) the lr_scheduler "no scheduler" guard tolerates
# CLI null forms; (2) the config lr is re-applied to param_groups after the
# optimizer-state load (which would otherwise restore the checkpoint's saved lr).
# Mirrors _setup_lr_scheduler (~1367) and _setup_optimizer (~2291) logic.
# ---------------------------------------------------------------------------


def _no_scheduler(cfg_lr_scheduler):
    # Mirror of the recipe's _no_sched guard.
    return (
        cfg_lr_scheduler is None
        or isinstance(cfg_lr_scheduler, str)
        or cfg_lr_scheduler.get("_component_", None) in (None, "None", "null")
    )


def test_scheduler_guard_tolerates_null_forms():
    # None, the string "null"/"None", and a node with _component_=None all mean
    # "constant lr" — must NOT try to instantiate (was "Invalid path: 'None'").
    assert _no_scheduler(None) is True
    assert _no_scheduler("null") is True
    assert _no_scheduler("None") is True
    assert _no_scheduler({"_component_": None}) is True
    assert _no_scheduler({"_component_": "None"}) is True
    # A real scheduler node must still instantiate.
    assert _no_scheduler(
        {"_component_": "torchtune.training.lr_schedulers.x", "num_warmup_steps": 10}
    ) is False


def _reapply_lr(param_groups, cfg_optimizer):
    # Mirror of the recipe's post-load lr re-apply.
    cfg_lr = cfg_optimizer.get("lr", None)
    if cfg_lr is not None:
        for pg in param_groups:
            pg["lr"] = float(cfg_lr)
    return param_groups


def test_resume_reapplies_config_lr_over_saved():
    # After optimizer.load_state_dict restores the checkpoint's lr (e.g. the
    # cosine-scheduled 4.17e-6 at step 150), the config lr=1e-5 must win so a
    # resume can change lr.
    restored = [{"lr": 4.17e-6}, {"lr": 4.17e-6}]
    out = _reapply_lr(restored, {"lr": 1e-5})
    assert all(pg["lr"] == 1e-5 for pg in out)


def test_resume_lr_noop_when_config_lr_absent():
    # If the optimizer config has no lr, leave the restored lr untouched.
    restored = [{"lr": 4.17e-6}]
    out = _reapply_lr(restored, {})
    assert out[0]["lr"] == 4.17e-6
