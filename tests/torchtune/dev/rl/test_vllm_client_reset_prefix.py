# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest

from torchtune.dev.rl.vllm_client import VLLMClient


def test_reset_prefix_cache_can_fail_fast():
    client = object.__new__(VLLMClient)
    client.base_url = "http://server"
    client.session = mock.Mock()
    client.session.post.return_value = mock.Mock(status_code=500, text="failed")

    with pytest.raises(RuntimeError, match="500 failed"):
        client.reset_prefix_cache(fail_on_error=True)

    client.session.post.assert_called_once_with(
        "http://server/reset_prefix_cache/", params=None, timeout=30
    )


def test_reset_prefix_cache_can_force_running_request_reset():
    client = object.__new__(VLLMClient)
    client.base_url = "http://server"
    client.session = mock.Mock()
    client.session.post.return_value = mock.Mock(status_code=200, text="")

    client.reset_prefix_cache(
        fail_on_error=True, reset_running_requests=True
    )

    client.session.post.assert_called_once_with(
        "http://server/reset_prefix_cache/",
        params={"reset_running_requests": "true"},
        timeout=30,
    )


def test_reset_prefix_cache_remains_best_effort_by_default():
    client = object.__new__(VLLMClient)
    client.base_url = "http://server"
    client.session = mock.Mock()
    client.session.post.side_effect = TimeoutError("timed out")

    client.reset_prefix_cache()
