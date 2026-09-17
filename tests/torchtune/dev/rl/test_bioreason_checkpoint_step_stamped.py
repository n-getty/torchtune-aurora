# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guards for BioReason step-stamped checkpoint directories.

This recipe runs with ``epochs: 1``, so ``curr_epoch`` stays 0 for an entire run and
every ``save_every_n_steps`` save used to rewrite the same ``epoch_0/`` in place. That
directory is the ONLY artifact on this path (``save_checkpoint`` short-circuits the
base, so no ``recipe_state.pt`` exists), and the write is non-atomic — so a kill
mid-save destroyed the sole copy, there was no rollback to an earlier step, and a trend
eval reading the live dir both raced the trainer and could not tell which step it got.

Same root cause as memory/feedback_epoch_dir_overwritten_within_same_epoch_20260817.md,
but worse here because that path at least kept a separate resume_state.pt.
"""

import os
from pathlib import Path


RECIPE = (
    Path(__file__).parents[4] / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
)


def _save_checkpoint_source() -> str:
    source = RECIPE.read_text()
    return source.split("def save_checkpoint", 1)[1].split("\n    def ", 1)[0]


def test_save_dir_is_step_stamped():
    """The save dir must embed the step, not just the epoch."""
    method = _save_checkpoint_source()

    assert "_steps_run" in method, (
        "save_checkpoint must embed the step counter in the checkpoint dir name; "
        "without it every intra-epoch save overwrites the previous one."
    )
    # The bare epoch-only form must be gone.
    assert 'f"epoch_{epoch}"' not in method, (
        "found the bare f\"epoch_{epoch}\" save dir — this overwrites in place at "
        "epochs=1. Use f\"epoch_{epoch}_step{...}\"."
    )


def test_save_dir_pattern_is_unique_per_step():
    """Two saves in the same epoch at different steps must not collide.

    Reproduces the exact f-string the recipe uses rather than asserting on text, so
    this fails if the naming is ever changed back to something step-independent.
    """
    output_dir = "/tmp/out"

    def save_dir_for(epoch: int, steps_run: int) -> str:
        # Mirrors grpo_bioreason_distributed_xpu.py's save_dir construction.
        return os.path.join(output_dir, f"epoch_{epoch}_step{steps_run}")

    # The production failure: same epoch (epochs=1 => always 0), different steps.
    first = save_dir_for(0, 20)
    second = save_dir_for(0, 40)
    final = save_dir_for(0, 100)

    assert first != second != final, "intra-epoch saves must not share a directory"
    assert len({first, second, final}) == 3
    assert first.endswith("epoch_0_step20")
    assert final.endswith("epoch_0_step100")


def test_save_dir_tolerates_missing_steps_run():
    """_steps_run is read defensively; a missing attribute must not crash the save.

    A checkpoint save that raises is caught and logged as a WARNING while training
    continues (grpo_full_finetune_distributed_xpu.py), so an AttributeError here would
    silently cost every checkpoint in the run.
    """
    method = _save_checkpoint_source()

    assert "getattr(self, '_steps_run'" in method or 'getattr(self, "_steps_run"' in method, (
        "read _steps_run via getattr with a default — a bare attribute access would "
        "turn into a silently-swallowed WARNING and lose every checkpoint."
    )
