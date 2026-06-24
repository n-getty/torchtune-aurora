"""Guard for the vLLM stop_token_ids fix (2026-06-23, BioReason gen cost).

ROOT CAUSE (4N bands A/B, job 8556851): generation was max_gen-bound, NOT
dispatch-bound. The recipe built `self._stop_token_ids` for post-hoc train-side
truncation but NEVER passed stop tokens to vLLM, so every rollout decoded to the
full max_tokens cap server-side (measured stop_rate=0.000, trunc_rate~0.5 — half
the sequences ran 1024 tokens needlessly). The straggler-band reshuffling could
not help because a 1024-token decode costs the same wall-clock regardless of how
seqs are spread across engines.

FIX: vllm_client.generate_from_embeds forwards an optional stop_token_ids into the
/v1/completions payload; the recipe passes self._stop_token_ids_list in BOTH
gen_kwargs blocks (sync HSDP path + async http path).

These are source/behavior guards (the real effect needs HW: stop_rate>0, shorter
mean response length, faster gen).
"""
import inspect
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
_CLIENT = _REPO / "torchtune" / "dev" / "rl" / "vllm_client.py"


def test_client_accepts_and_forwards_stop_token_ids():
    src = _CLIENT.read_text()
    # signature param
    assert "stop_token_ids" in src
    # it must land in the JSON payload, not just be accepted and dropped
    assert 'payload["stop_token_ids"]' in src


def test_recipe_builds_stop_token_id_list():
    src = _RECIPE.read_text()
    assert "_stop_token_ids_list" in src
    # built from the same stop_tokens source as the tensor form
    assert "int(t) for t in stop_token_ids" in src


def test_both_gen_kwargs_pass_stop_token_ids():
    src = _RECIPE.read_text()
    # both the sync (_generate_with_vllm_server_embeds) and async
    # (_http_generate_from_embeds_cpu) gen_kwargs blocks must include it.
    n = src.count("stop_token_ids=getattr(self, \"_stop_token_ids_list\", None)")
    assert n >= 2, f"expected stop_token_ids in both gen_kwargs blocks, found {n}"


def test_client_signature_has_stop_token_ids_param():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_vc", _CLIENT)
    # We can't import (pulls torch/vllm deps); inspect the source signature instead.
    src = _CLIENT.read_text()
    # generate_from_embeds def block must list stop_token_ids before the return type.
    blk = src.split("def generate_from_embeds", 1)[1].split(")", 1)[0]
    assert "stop_token_ids" in blk


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
