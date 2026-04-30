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
