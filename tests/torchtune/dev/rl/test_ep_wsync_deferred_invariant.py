# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Static-config invariant: EP MoE configs that enable XCCL vLLM weight sync
must opt OUT of the deferred-broadcast schedule.

WS3 (2026-05-01, hold 8464081) discovered that the default deferred broadcast
in `_sync_weights_to_vllm_xccl` (weight_sync.py:1834-1856) fires the actual
XCCL broadcast during step N+1's grad-release window. For dense FSDP this is
fine — dp_shard is gloo or unused during gen. For EP modes, the broadcast
contends with `_ep_release_fsdp_unsharded_grads` on the same dp_shard XCCL
fabric and deterministically hangs.

The synchronous mode (vllm_weight_sync_deferred=false) runs the broadcast
inline at end-of-step, costs ~7-15s of unhidden wall time, but is correct
for EP. See docs/reports/MoE_EP_status.md section #5 for the design history.
"""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

CFG_ROOT = (
    Path(__file__).resolve().parents[4] / "recipes" / "configs" / "dev"
)


def _all_yamls():
    return sorted(CFG_ROOT.rglob("*.yaml"))


@pytest.mark.parametrize("yaml_path", _all_yamls(), ids=lambda p: p.name)
def test_ep_xccl_wsync_must_be_synchronous(yaml_path: Path):
    cfg = OmegaConf.load(str(yaml_path))

    ep_degree = int(cfg.get("expert_parallel_degree", 1) or 1)
    if ep_degree <= 1:
        pytest.skip(f"{yaml_path.name}: not an EP config (ep_degree={ep_degree})")

    if not bool(cfg.get("vllm_weight_sync", False)):
        pytest.skip(f"{yaml_path.name}: vllm_weight_sync disabled")

    method = cfg.get("vllm_weight_sync_method", "raw_bytes")
    if method != "xccl":
        pytest.skip(f"{yaml_path.name}: wsync method={method} (deferred only applies to xccl)")

    deferred = cfg.get("vllm_weight_sync_deferred", True)
    assert deferred is False, (
        f"{yaml_path.name}: expert_parallel_degree={ep_degree} with "
        f"vllm_weight_sync=true and method=xccl requires "
        "vllm_weight_sync_deferred: false. The default deferred broadcast "
        "fires during step N+1 grad-release on the dp_shard XCCL fabric, "
        "which contends with `_ep_release_fsdp_unsharded_grads` and hangs. "
        "See docs/reports/MoE_EP_status.md section #5."
    )
