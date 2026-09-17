# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


REPO = Path(__file__).parents[4]


def test_bioreason_enables_nonfinite_gradient_skip_by_default():
    source = (REPO / "recipes/dev/grpo_bioreason_distributed_xpu.py").read_text()
    setup = source.split("def setup", 1)[1].split("def ", 1)[0]

    assert 'cfg.get("skip_nonfinite_grad_step", True)' in setup


def test_train_collectively_skips_nonfinite_gradient_update():
    source = (REPO / "recipes/dev/grpo_full_finetune_distributed_xpu.py").read_text()
    train = source.split("def train", 1)[1]

    assert "torch.isfinite(grad_norm).all()" in train
    assert "op=torch.distributed.ReduceOp.MIN" in train
    assert '"NONFINITE_GRAD step=%d' in train
    assert "if _skip_zero_adv or _skip_nonfinite_grad:" in train
