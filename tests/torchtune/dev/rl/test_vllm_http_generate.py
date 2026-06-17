# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe tests for the shared ``vllm_http_generate`` helper.

This helper was extracted from the dense GRPO recipe's ``_call_vllm_http`` so
the dense and LoRA-GRPO recipes share one implementation (the LoRA fork's old
inline copy was missing prompt truncation, stop-token/stop-string forwarding,
and EOS-injection). These tests pin the contract that both recipes now depend
on, using stub clients — no XPU, no HTTP, no vLLM.
"""
from __future__ import annotations

import unittest

import torch

from torchtune.dev.rl.vllm_client import vllm_http_generate


class _StubClient:
    """Records the prompts/kwargs it was called with; returns canned tokens.

    ``completions_for(prompt)`` maps each prompt (tuple) to its completion so we
    can verify round-robin reassembly is correct regardless of client.
    """

    def __init__(self, table):
        self._table = table  # dict[tuple[int,...], list[int]]
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": [list(p) for p in prompts], "kwargs": kwargs})
        return [list(self._table[tuple(p)]) for p in prompts]


PAD = 0
EOS = 99


class TestVllmHttpGenerate(unittest.TestCase):
    def _ids(self, rows):
        return torch.tensor(rows, dtype=torch.long)

    def test_single_client_basic_layout(self):
        # 2 prompts, context_length=3, max_gen=4.
        batch = self._ids([[1, 2, 3], [4, 5, 6]])
        table = {(1, 2, 3): [11, 12], (4, 5, 6): [21, 22, 23]}
        client = _StubClient(table)
        out = vllm_http_generate(
            batch, context_length=3,
            vllm_clients=[client], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=4, vllm_max_model_len=64,
            temperature=0.7, top_k=0,
        )
        self.assertEqual(out.shape, (2, 7))  # 3 + 4
        # Prompt preserved.
        self.assertEqual(out[0, :3].tolist(), [1, 2, 3])
        # Completion placed after context.
        self.assertEqual(out[0, 3:5].tolist(), [11, 12])
        # EOS injected at first pad after a short completion (len 2 < max 4).
        self.assertEqual(out[0, 5].item(), EOS)
        self.assertEqual(out[1, 3:6].tolist(), [21, 22, 23])
        self.assertEqual(out[1, 6].item(), EOS)

    def test_no_eos_injection_when_completion_fills_max(self):
        batch = self._ids([[1, 2]])
        table = {(1, 2): [7, 8, 9]}  # length == max_generated_tokens (3)
        out = vllm_http_generate(
            batch, context_length=2,
            vllm_clients=[_StubClient(table)], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=3, vllm_max_model_len=64,
            temperature=0.0, top_k=0,
        )
        # All 3 slots filled by completion; no room for injected EOS.
        self.assertEqual(out[0, 2:5].tolist(), [7, 8, 9])
        self.assertNotIn(EOS, out[0].tolist())

    def test_eos_none_disables_injection(self):
        batch = self._ids([[1, 2]])
        table = {(1, 2): [7]}
        out = vllm_http_generate(
            batch, context_length=2,
            vllm_clients=[_StubClient(table)], pad_id=PAD, eos_id=None,
            max_generated_tokens=3, vllm_max_model_len=64,
            temperature=0.0, top_k=0,
        )
        self.assertEqual(out[0, 2].item(), 7)
        # Remaining positions stay pad (no EOS).
        self.assertEqual(out[0, 3:].tolist(), [PAD, PAD])

    def test_padding_stripped_from_prompts(self):
        # Row has trailing pad that must not be sent to the client.
        batch = self._ids([[1, 2, PAD]])
        table = {(1, 2): [5]}
        client = _StubClient(table)
        vllm_http_generate(
            batch, context_length=3,
            vllm_clients=[client], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=2, vllm_max_model_len=64,
            temperature=0.0, top_k=0,
        )
        self.assertEqual(client.calls[0]["prompts"], [[1, 2]])

    def test_prompt_left_truncated_to_model_len_budget(self):
        # vllm_max_model_len=5, max_gen=2 -> max_prompt_len=3; keep LAST 3.
        batch = self._ids([[1, 2, 3, 4, 5]])
        table = {(3, 4, 5): [6]}
        client = _StubClient(table)
        vllm_http_generate(
            batch, context_length=5,
            vllm_clients=[client], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=2, vllm_max_model_len=5,
            temperature=0.0, top_k=0,
        )
        self.assertEqual(client.calls[0]["prompts"], [[3, 4, 5]])

    def test_stop_kwargs_forwarded(self):
        batch = self._ids([[1, 2]])
        table = {(1, 2): [3]}
        client = _StubClient(table)
        vllm_http_generate(
            batch, context_length=2,
            vllm_clients=[client], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=3, vllm_max_model_len=64,
            temperature=0.5, top_k=7,
            stop_token_ids=torch.tensor([99, 100]),
            stop_strings=["</answer>", "User:"],
        )
        kw = client.calls[0]["kwargs"]
        self.assertEqual(kw["stop_token_ids"], [99, 100])
        self.assertEqual(kw["stop"], ["</answer>", "User:"])
        self.assertEqual(kw["top_k"], 7)
        self.assertEqual(kw["temperature"], 0.5)
        self.assertEqual(kw["max_tokens"], 3)

    def test_stop_kwargs_omitted_when_unset(self):
        batch = self._ids([[1, 2]])
        table = {(1, 2): [3]}
        client = _StubClient(table)
        vllm_http_generate(
            batch, context_length=2,
            vllm_clients=[client], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=3, vllm_max_model_len=64,
            temperature=0.5, top_k=None,
        )
        kw = client.calls[0]["kwargs"]
        self.assertNotIn("stop_token_ids", kw)
        self.assertNotIn("stop", kw)
        self.assertEqual(kw["top_k"], 0)  # None -> 0

    def test_multi_client_round_robin_reassembly(self):
        # 4 prompts across 2 clients; prompt i -> client i%2, within i//2.
        # The helper must reassemble completions back to original prompt order.
        batch = self._ids([[1], [2], [3], [4]])
        # Each client returns the completion keyed by the prompt it actually got.
        table = {(1,): [10], (2,): [20], (3,): [30], (4,): [40]}
        c0 = _StubClient(table)
        c1 = _StubClient(table)
        out = vllm_http_generate(
            batch, context_length=1,
            vllm_clients=[c0, c1], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=2, vllm_max_model_len=64,
            temperature=0.0, top_k=0,
        )
        # client 0 gets prompts [1,3] (i=0,2), client 1 gets [2,4] (i=1,3).
        self.assertEqual(c0.calls[0]["prompts"], [[1], [3]])
        self.assertEqual(c1.calls[0]["prompts"], [[2], [4]])
        # Reassembled in original order: completions 10,20,30,40.
        self.assertEqual(out[0, 1].item(), 10)
        self.assertEqual(out[1, 1].item(), 20)
        self.assertEqual(out[2, 1].item(), 30)
        self.assertEqual(out[3, 1].item(), 40)

    def test_completion_truncated_to_max_generated_tokens(self):
        # Client returns MORE tokens than max_generated_tokens; must clip.
        batch = self._ids([[1]])
        table = {(1,): [5, 6, 7, 8, 9]}
        out = vllm_http_generate(
            batch, context_length=1,
            vllm_clients=[_StubClient(table)], pad_id=PAD, eos_id=EOS,
            max_generated_tokens=2, vllm_max_model_len=64,
            temperature=0.0, top_k=0,
        )
        self.assertEqual(out.shape, (1, 3))  # 1 + 2
        self.assertEqual(out[0, 1:3].tolist(), [5, 6])


if __name__ == "__main__":
    unittest.main()
