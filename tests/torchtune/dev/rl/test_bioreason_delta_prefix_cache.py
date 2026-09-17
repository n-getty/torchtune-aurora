# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


WEIGHT_SYNC = (
    Path(__file__).parents[4] / "torchtune" / "dev" / "rl" / "weight_sync.py"
)


def test_bioreason_delta_publish_invalidates_prefix_cache_after_weight_update():
    source = WEIGHT_SYNC.read_text()
    method = source.split("def _publish_bioreason_lora_delta", 1)[1].split(
        "def _xccl_gather_fsdp1", 1
    )[0]

    publish = method.index("_post_bioreason_collective_rpc(")
    invalidate = method.index("client.reset_prefix_cache(")
    release = method.index("torch.distributed.broadcast_object_list(")

    assert invalidate > publish
    assert "for client in self._vllm_clients:" in method[publish:invalidate]
    assert release > invalidate
    assert "if not _is_xccl_leader:" in method[release:]

    # reset_running_requests used to be the literal `True` here. It is now read
    # from TORCHTUNE_WSYNC_RESET_RUNNING_REQUESTS so the async-overlap probe can
    # turn it off: under async, the publish fires while the RolloutProducer's
    # next request is mid-decode, and preempting it re-prefills exactly the work
    # the overlap exists to hide. The DEFAULT must stay "1" so this synchronous
    # production path is bit-for-bit unchanged, which is what this now asserts.
    gate = method[invalidate - 400 : invalidate]
    assert "reset_running_requests=_reset_running" in method[invalidate:release]
    assert 'os.environ.get("TORCHTUNE_WSYNC_RESET_RUNNING_REQUESTS", "1")' in gate
