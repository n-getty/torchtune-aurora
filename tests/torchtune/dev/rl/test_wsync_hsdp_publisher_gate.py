"""CPU-safe guard for the HSDP-safe weight-sync publisher gate (raw_bytes path).

Regression (2026-06-22, found wiring BioReason HSDP): the server-mode raw_bytes
weight sync gated the save-to-_weight_sync_path + HTTP POST on `self._is_shard_leader`.
That is correct for a single replica (shard leader == global rank 0), but with HSDP
(`data_parallel_replicate_dim > 1`) there are MULTIPLE shard leaders — one per
replica. Each would save to the SAME _weight_sync_path file and POST to the SAME
shared vLLM pool → file clobber + racing requests. All replicas are weight-identical
(HYBRID_SHARD all-reduces grads across the replicate dim), so exactly ONE rank
(global rank 0) must publish.

Fix: a `_is_publisher` gate = (rank == 0) when dp_replicate>1 else _is_shard_leader.
The FULL_STATE_DICT gather stays COLLECTIVE (all shard ranks call state_dict()); only
the populate + save + POST are publisher-gated.

This test pins the gate at the source level (the real check needs a live multi-rank
FSDP+vLLM stack). It fails if the save/POST reverts to a bare _is_shard_leader gate.
"""
import ast
from pathlib import Path

WEIGHT_SYNC = (
    Path(__file__).resolve().parents[4]
    / "torchtune" / "dev" / "rl" / "weight_sync.py"
)


def _func_src(name: str) -> str:
    src = WEIGHT_SYNC.read_text()
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    assert fn is not None, f"{name} not found in weight_sync.py"
    return ast.get_source_segment(src, fn)


def test_raw_bytes_publisher_gate_is_hsdp_safe():
    fn = _func_src("_sync_weights_to_vllm")
    # The HSDP-safe publisher predicate must exist and key off dp_replicate.
    assert "_is_publisher" in fn, (
        "raw_bytes wsync must define an _is_publisher gate (HSDP: only global rank 0 "
        "publishes when dp_replicate>1, else the shard leader)"
    )
    assert "_dp_replicate" in fn and "self.rank == 0" in fn, (
        "_is_publisher must select global rank 0 when _dp_replicate>1"
    )
    # The save+POST must NOT be gated on a bare _is_shard_leader anymore (that fired
    # on every replica's leader → file clobber + racing POSTs to the shared vLLM pool).
    # The collective state_dict() gather may still be unconditional; what matters is
    # the save/POST guard. Assert the publisher gate is used for the save block.
    assert fn.count("if _is_publisher:") >= 2, (
        "both the state-dict populate AND the save+POST must gate on _is_publisher"
    )


def test_collective_state_dict_stays_unconditional():
    """The FULL_STATE_DICT gather is a collective — all shard ranks must call
    state_dict() (it must NOT be moved inside the publisher gate, or non-publisher
    ranks would skip the collective and the gather would hang)."""
    fn = _func_src("_sync_weights_to_vllm")
    # state_dict() call appears, and the line immediately guarding it is the
    # state_dict_type context, NOT an `if _is_publisher:`. Cheap structural check:
    assert "full_sd = self._model.state_dict()" in fn
    # Ensure the populate-into-hf_state_dict (publisher-gated) comes AFTER the
    # collective gather (so the gather is unconditional).
    gather_idx = fn.index("full_sd = self._model.state_dict()")
    pub_idx = fn.index("if _is_publisher:")
    assert gather_idx < pub_idx, (
        "the collective state_dict() gather must precede (and be outside) the "
        "_is_publisher gate so all ranks participate"
    )


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
