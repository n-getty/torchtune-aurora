# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test: bind LoRAGRPODistributedXPU.grpo_step to a fake recipe.

Why source extraction instead of importing the recipe module:
  The recipe module pulls in torchao + distributed + XPU backends at import
  time; on the login node torchao crashes with std::bad_alloc.  We can't
  import the class to test it.

  Instead, parse the source file, extract the grpo_step method, and exec it
  in a controlled namespace so the test exercises the ACTUAL bytes shipped
  in the recipe.  Any regression (drop the ~, swap masks, change the
  loss-call shape) fails the test.
"""
import ast
import textwrap
import unittest

import torch


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "lora_grpo_full_finetune_distributed_xpu.py"
)


def _extract_grpo_step_source() -> str:
    """Pull `def grpo_step(...)` source out of the recipe file."""
    with open(_RECIPE_PATH) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LoRAGRPODistributedXPU":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "grpo_step":
                    return textwrap.dedent(ast.get_source_segment(open(_RECIPE_PATH).read(), item))
    raise RuntimeError("Could not find LoRAGRPODistributedXPU.grpo_step")


class _FakeLoss(torch.nn.Module):
    """Records every call's `padding_masks` arg + returns differentiable scalars."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, old_lp, pi_lp, ref_lp, adv, padding_masks=None):
        self.calls.append({
            "padding_masks": padding_masks.clone() if padding_masks is not None else None,
            "old_shape": tuple(old_lp.shape),
            "pi_shape": tuple(pi_lp.shape),
        })
        # Differentiable loss tied to pi_lp so backward() doesn't error.
        loss = pi_lp.sum() * 0.0 + 1.0
        scalar = loss.detach()
        return loss, scalar, scalar, scalar, scalar


class _FakeModel(torch.nn.Module):
    """Tiny passthrough returning logits with grad enabled."""

    def __init__(self, vocab_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, query_responses, input_pos=None, mask=None):
        B, S = query_responses.shape
        return torch.zeros(B, S, self.vocab_size) + self.w

    def train(self, mode: bool = True):
        return super().train(mode)


def _make_trajectory(B=2, prompt_len=4, resp_len=6, n_pad=2):
    from torchtune.dev.rl.types import GRPOTrajectory
    S = prompt_len + resp_len
    query_responses = torch.randint(0, 128, (B, S))
    response_padding_masks = torch.zeros(B, resp_len, dtype=torch.bool)
    response_padding_masks[:, -n_pad:] = True
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    logprobs = torch.randn(B, resp_len)
    ref_logprobs = torch.randn(B, resp_len)
    advantages = torch.randn(B)
    return GRPOTrajectory(
        query_responses=query_responses,
        logprobs=logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        rewards=torch.zeros(B),
        successes=torch.zeros(B),
        masks=None,
        position_ids=position_ids,
        response_padding_masks=response_padding_masks,
        seq_lens=None,
    )


class _FakeOptimizer:
    def __init__(self, params):
        self._params = list(params)

    def zero_grad(self, set_to_none: bool = True):
        for p in self._params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()


def _build_recipe_stub(model, loss, fwd_bs=2, grad_accum=1):
    """Minimal namespace with every attribute grpo_step touches."""
    class Stub:
        pass
    stub = Stub()
    stub._model = model
    stub._loss_fn = loss
    stub._optimizer = _FakeOptimizer(model.parameters())
    stub._forward_batch_size = fwd_bs
    stub._temperature = 1.0
    stub._gradient_accumulation_steps = grad_accum
    stub._device = torch.device("cpu")
    stub._is_rank_zero = False
    return stub


def _bind_grpo_step():
    """Compile the extracted source and return the function object."""
    src = _extract_grpo_step_source()
    # The method references module-globals (rlhf, log, os, torch).  Provide them.
    import os as _os
    import logging as _logging
    import torchtune.rlhf as rlhf  # has truncate_sequence_for_logprobs + batched_logits_to_logprobs
    from torchtune.dev.rl.types import GRPOTrajectory
    ns = {
        "torch": torch,
        "rlhf": rlhf,
        "os": _os,
        "log": _logging.getLogger("test_lora_grpo_step_mask"),
        "GRPOTrajectory": GRPOTrajectory,
    }
    exec(compile(src, _RECIPE_PATH, "exec"), ns)
    return ns["grpo_step"]


class TestLoRAGrpoStepBinding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grpo_step = staticmethod(_bind_grpo_step())

    def test_single_fwd_mask_polarity(self):
        """Non-chunked path passes ~response_padding_masks (True=valid token)."""
        model = _FakeModel()
        loss = _FakeLoss()
        # fwd_bs >= num_seqs -> non-chunked path
        recipe = _build_recipe_stub(model, loss, fwd_bs=8, grad_accum=1)
        traj = _make_trajectory(B=2, prompt_len=4, resp_len=6, n_pad=2)
        # Force non-chunked code path
        import os
        os.environ.pop("TORCHTUNE_USE_CHUNKED_LOSS", None)
        self.grpo_step(recipe, traj)
        self.assertEqual(len(loss.calls), 1)
        received = loss.calls[0]["padding_masks"]
        expected = ~traj.response_padding_masks
        self.assertTrue(
            torch.equal(received, expected),
            f"Non-chunked grpo_step must pass ~response_padding_masks. "
            f"Got polarity match={torch.equal(received, traj.response_padding_masks)} (means recipe dropped the ~).",
        )

    def test_chunked_mask_polarity_and_call_count(self):
        """Chunked path passes ~response_padding_masks[cs:ce] for each chunk."""
        model = _FakeModel()
        loss = _FakeLoss()
        recipe = _build_recipe_stub(model, loss, fwd_bs=2, grad_accum=1)
        traj = _make_trajectory(B=4, prompt_len=4, resp_len=6, n_pad=2)
        import os
        os.environ["TORCHTUNE_USE_CHUNKED_LOSS"] = "1"
        try:
            self.grpo_step(recipe, traj)
        finally:
            os.environ.pop("TORCHTUNE_USE_CHUNKED_LOSS", None)
        # 4 / 2 = 2 chunks
        self.assertEqual(len(loss.calls), 2, f"expected 2 chunks, got {len(loss.calls)}")
        for i, call in enumerate(loss.calls):
            cs = i * 2
            ce = cs + 2
            expected = ~traj.response_padding_masks[cs:ce]
            self.assertTrue(
                torch.equal(call["padding_masks"], expected),
                f"chunk {i}: mask polarity wrong (recipe likely dropped the ~).",
            )

    def test_grpo_step_returns_loss_and_kl(self):
        """grpo_step returns dict with 'loss' and 'kl' float keys."""
        model = _FakeModel()
        loss = _FakeLoss()
        recipe = _build_recipe_stub(model, loss, fwd_bs=8, grad_accum=1)
        traj = _make_trajectory()
        out = self.grpo_step(recipe, traj)
        self.assertIsInstance(out, dict)
        self.assertIn("loss", out)
        self.assertIn("kl", out)
        self.assertIsInstance(out["loss"], float)
        self.assertIsInstance(out["kl"], float)


if __name__ == "__main__":
    unittest.main()
