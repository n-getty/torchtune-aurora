# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe tests for prompt-embedding generation request failure handling."""
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


def test_session_adapter_does_not_retry_post(monkeypatch):
    monkeypatch.setattr(VLLMClient, "check_server", lambda self, timeout: None)
    client = VLLMClient("http://localhost:8001")

    retries = client.session.get_adapter("http://").max_retries

    assert "POST" not in retries.allowed_methods
    assert "GET" in retries.allowed_methods


def test_does_not_replay_ambiguous_generation_failure(monkeypatch):
    client = _make_client()
    client.session.post.side_effect = requests.exceptions.ConnectionError("timed out")
    health = _mock_response(status=200)
    monkeypatch.setattr(requests, "get", MagicMock(return_value=health))

    try:
        client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "not replayed" in str(e)
        assert "health=HTTP 200" in str(e)

    assert client.session.post.call_count == 1


def test_reports_failed_health_probe(monkeypatch):
    client = _make_client()
    client.session.post.side_effect = requests.exceptions.ConnectionError("timed out")
    monkeypatch.setattr(
        requests,
        "get",
        MagicMock(side_effect=requests.exceptions.ConnectTimeout("dead")),
    )

    try:
        client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "health=unreachable" in str(e)

    assert client.session.post.call_count == 1


def test_succeeds_immediately_when_no_transient_failure():
    client = _make_client()
    client.session.post.return_value = _mock_response(
        json_data={"choices": [{"token_ids": [4, 5]}]}
    )

    out = client.generate_from_embeds(prompt_embeds=_one_embed(), max_tokens=16)

    assert out == [[4, 5]]
    assert client.session.post.call_count == 1
    assert client.session.post.call_args.kwargs["timeout"] == 3600.0


def test_multiple_samples_forward_n_and_preserve_choice_order():
    client = _make_client()
    client.session.post.return_value = _mock_response(
        json_data={
            "choices": [
                {"index": 0, "token_ids": [10]},
                {"index": 1, "token_ids": [11]},
                {"index": 2, "token_ids": [20]},
                {"index": 3, "token_ids": [21]},
            ]
        }
    )

    out = client.generate_from_embeds(
        prompt_embeds=[
            torch.zeros(4, 8, dtype=torch.bfloat16),
            torch.ones(4, 8, dtype=torch.bfloat16),
        ],
        n=2,
        max_tokens=16,
    )

    assert out == [[10], [11], [20], [21]]
    assert client.session.post.call_args.kwargs["json"]["n"] == 2
    assert client.session.post.call_args.kwargs["json"]["return_token_ids"] is True


def test_prompt_embed_rows_serialize_only_logical_storage():
    client = _make_client()
    expanded = (
        torch.zeros(2, 1, 4, 8, dtype=torch.bfloat16)
        .expand(2, 8, 4, 8)
        .reshape(16, 4, 8)
        .contiguous()
    )
    client.session.post.return_value = _mock_response(
        json_data={"choices": [{"token_ids": [index]} for index in range(16)]}
    )

    client.generate_from_embeds(
        prompt_embeds=[expanded[index] for index in range(16)], max_tokens=16
    )

    encoded = client.session.post.call_args.kwargs["json"]["prompt_embeds"]
    loaded = torch.load(io.BytesIO(base64.b64decode(encoded[0])), weights_only=True)
    assert loaded.untyped_storage().nbytes() == loaded.numel() * loaded.element_size()


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
