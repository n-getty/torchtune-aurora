# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test for generate_from_embeds' retry-on-transient-failure logic.

At high fan-in (e.g. 16-node BioReason GRPO, 240 concurrent /v1/completions
requests/step against a 2-engine centralized vLLM pool), a single request can
hit a transient connection timeout even though the vLLM server itself is
healthy (confirmed via pbsnodes: not an offline/dead node). Before this fix,
``generate_from_embeds`` had no retry — one such timeout raised immediately,
killing the calling rank and, via MPI's default fail-fast behavior, the entire
multi-hundred-rank job (observed on job 8812775, rank 72, MaxRetryError).
"""
import base64
import io
from unittest.mock import MagicMock

import requests
import torch

from torchtune.dev.rl.vllm_client import VLLMClient


def _make_client() -> VLLMClient:
    client = VLLMClient.__new__(VLLMClient)
    client.session = MagicMock()
    client.base_url = "http://localhost:8001"
    client._api_type = "openai"
    client._model_name = "test-model"
    return client


def _mock_response(status: int = 200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    return resp


def _one_embed():
    return [torch.zeros(4, 8, dtype=torch.bfloat16)]


def test_retries_on_transient_connection_error_then_succeeds():
    client = _make_client()
    ok_response = _mock_response(json_data={"choices": [{"token_ids": [1, 2, 3]}]})
    client.session.post.side_effect = [
        requests.exceptions.ConnectionError("timed out"),
        ok_response,
    ]

    out = client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)

    assert out == [[1, 2, 3]]
    assert client.session.post.call_count == 2


def test_raises_after_exhausting_all_retry_attempts():
    client = _make_client()
    client.session.post.side_effect = requests.exceptions.ConnectionError("timed out")

    try:
        client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "3 attempts" in str(e)

    assert client.session.post.call_count == 3


def test_succeeds_immediately_when_no_transient_failure():
    client = _make_client()
    client.session.post.return_value = _mock_response(
        json_data={"choices": [{"token_ids": [4, 5]}]}
    )

    out = client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)

    assert out == [[4, 5]]
    assert client.session.post.call_count == 1


def test_does_not_retry_on_non_200_status_code():
    """A real vLLM-side error (bad request, OOM) should fail fast, not retry —
    only network-level exceptions (timeouts, connection resets) are transient."""
    client = _make_client()
    client.session.post.return_value = _mock_response(status=500, json_data={})
    client.session.post.return_value.text = "internal server error"

    try:
        client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "500" in str(e)

    assert client.session.post.call_count == 1
