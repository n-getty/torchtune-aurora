# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Static-config invariant: async_generation flag, loss class, and
always_compute_rollout_logprobs must agree.

The behavior policy for async generation is the vLLM rollout — its logprobs
are needed by GRPOLoss to form the IS ratio. If async is OFF, GRPOSimpleLoss
collapses the ratio to 1.0 and computing rollout logprobs is wasted work.

Invariants enforced:
  * async_generation.enabled == True  → loss must be GRPOLoss
                                       AND always_compute_rollout_logprobs == True
  * async_generation.enabled == False → always_compute_rollout_logprobs must be False
                                       (GRPOSimpleLoss is the natural choice; we
                                        don't enforce the loss class for sync runs
                                        because some sync configs intentionally use
                                        GRPOLoss for multi-epoch off-policy reuse)
"""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PROD_DIR = Path(__file__).resolve().parents[4] / "recipes" / "configs" / "dev" / "production"


def _all_prod_yamls():
    return sorted(PROD_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_path", _all_prod_yamls(), ids=lambda p: p.name)
def test_async_generation_loss_logprob_consistency(yaml_path: Path):
    cfg = OmegaConf.load(str(yaml_path))

    async_cfg = cfg.get("async_generation", None)
    if async_cfg is None:
        pytest.skip(f"{yaml_path.name} has no async_generation block")

    enabled = bool(async_cfg.get("enabled", False))
    always_logp = bool(cfg.get("always_compute_rollout_logprobs", False))

    loss_cfg = cfg.get("loss", None)
    loss_component = (
        loss_cfg.get("_component_") if loss_cfg is not None else None
    )

    if enabled:
        assert loss_component == "torchtune.dev.rl.loss.GRPOLoss", (
            f"{yaml_path.name}: async_generation.enabled=true requires "
            f"GRPOLoss (got {loss_component}). GRPOSimpleLoss collapses the "
            "IS ratio to 1.0 — async behavior-policy correction is lost."
        )
        assert always_logp, (
            f"{yaml_path.name}: async_generation.enabled=true requires "
            "always_compute_rollout_logprobs=true so the policy logprobs "
            "needed for the IS ratio are computed every step."
        )
    else:
        assert not always_logp, (
            f"{yaml_path.name}: async_generation.enabled=false but "
            "always_compute_rollout_logprobs=true — this wastes a full "
            "policy forward each step (GRPOSimpleLoss ignores the result)."
        )
