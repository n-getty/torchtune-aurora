# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test for the vLLM HTTP stop-token plumbing.

Before this fix, `VLLMClient.generate` had no `stop_token_ids` parameter and the
recipe's `_call_vllm_http` never told the vLLM server which token IDs should
terminate generation. Every completion ran to `max_tokens`, producing flat
`response_lengths` and a useless RL signal for any model that emits EOS.

Both API paths must forward a non-empty `stop_token_ids` into the request
payload, and must omit it (or leave it falsy) when not given, so that the
default behavior of all existing callers is preserved.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from torchtune.dev.rl.vllm_client import VLLMClient


def _make_client(api_type: str) -> VLLMClient:
    """Construct a VLLMClient without going through __init__ / network."""
    client = VLLMClient.__new__(VLLMClient)
    client.session = MagicMock()
    client.base_url = "http://localhost:8001"
    client._api_type = api_type
    client._model_name = "test-model"
    return client


def _mock_post_response(status: int = 200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    return resp


# ----- TRL path ---------------------------------------------------------------


def test_trl_payload_includes_stop_token_ids_when_provided():
    client = _make_client("trl")
    client.session.post.return_value = _mock_post_response(
        json_data={"completion_ids": [[1, 2, 3]]}
    )

    client.generate(
        prompts=[[10, 20]],
        n=1,
        max_tokens=64,
        temperature=0.7,
        top_k=0,
        stop_token_ids=[2, 100257],
    )

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert payload["stop_token_ids"] == [2, 100257]


def test_trl_payload_omits_stop_token_ids_when_none():
    client = _make_client("trl")
    client.session.post.return_value = _mock_post_response(
        json_data={"completion_ids": [[1]]}
    )

    client.generate(prompts=[[10]], n=1, max_tokens=8, temperature=0.7, top_k=0)

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert "stop_token_ids" not in payload


def test_trl_payload_omits_stop_token_ids_when_empty_list():
    """Empty list is falsy; preserve server-side default behavior."""
    client = _make_client("trl")
    client.session.post.return_value = _mock_post_response(
        json_data={"completion_ids": [[1]]}
    )

    client.generate(
        prompts=[[10]], n=1, max_tokens=8, temperature=0.7, top_k=0, stop_token_ids=[]
    )

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert "stop_token_ids" not in payload


# ----- OpenAI path ------------------------------------------------------------


def test_openai_payload_includes_stop_token_ids_when_provided():
    client = _make_client("openai")
    client.session.post.return_value = _mock_post_response(
        json_data={"choices": [{"token_ids": [1, 2, 3]}]}
    )

    client.generate(
        prompts=[[10, 20]],
        n=1,
        max_tokens=64,
        temperature=0.7,
        top_k=0,
        stop_token_ids=[2],
    )

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert payload["stop_token_ids"] == [2]


def test_openai_payload_omits_stop_token_ids_when_none():
    client = _make_client("openai")
    client.session.post.return_value = _mock_post_response(
        json_data={"choices": [{"token_ids": [1]}]}
    )

    client.generate(prompts=[[10]], n=1, max_tokens=8, temperature=0.7, top_k=0)

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert "stop_token_ids" not in payload


# ----- String stops (covers both API types) -----------------------------------


@pytest.mark.parametrize("api_type", ["trl", "openai"])
def test_payload_includes_string_stops_when_provided(api_type):
    """Raw pretraining checkpoints never emit EOS; ``stop`` is the only real
    termination lever. Required to be forwarded verbatim AND to set
    include_stop_str_in_output so regex-based reward extractors still match.
    """
    client = _make_client(api_type)
    json_body = (
        {"completion_ids": [[1]]}
        if api_type == "trl"
        else {"choices": [{"token_ids": [1]}]}
    )
    client.session.post.return_value = _mock_post_response(json_data=json_body)

    client.generate(
        prompts=[[10]],
        n=1,
        max_tokens=8,
        temperature=0.7,
        top_k=0,
        stop=["</answer>", "User:"],
    )

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert payload["stop"] == ["</answer>", "User:"]
    assert payload["include_stop_str_in_output"] is True


@pytest.mark.parametrize("api_type", ["trl", "openai"])
def test_payload_omits_stop_when_none(api_type):
    client = _make_client(api_type)
    json_body = (
        {"completion_ids": [[1]]}
        if api_type == "trl"
        else {"choices": [{"token_ids": [1]}]}
    )
    client.session.post.return_value = _mock_post_response(json_data=json_body)

    client.generate(prompts=[[10]], n=1, max_tokens=8, temperature=0.7, top_k=0)

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert "stop" not in payload
    assert "include_stop_str_in_output" not in payload


@pytest.mark.parametrize("api_type", ["trl", "openai"])
def test_payload_omits_stop_when_empty_list(api_type):
    client = _make_client(api_type)
    json_body = (
        {"completion_ids": [[1]]}
        if api_type == "trl"
        else {"choices": [{"token_ids": [1]}]}
    )
    client.session.post.return_value = _mock_post_response(json_data=json_body)

    client.generate(
        prompts=[[10]], n=1, max_tokens=8, temperature=0.7, top_k=0, stop=[]
    )

    _, kwargs = client.session.post.call_args
    payload = kwargs["json"]
    assert "stop" not in payload
    assert "include_stop_str_in_output" not in payload
