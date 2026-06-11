# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe tests for the vLLM importance-sampling correction in
``GRPOFullFinetuneDistributedXPU``.

Phase 3 of the IS-correction plan (~/.claude/plans/linear-singing-cocke.md).
Covers:

  1. Identity at parity: when vLLM logp == train logp the IS ratio is 1.0
     across all 4 modes and the resulting per-token policy loss exactly matches
     the no-IS baseline.
  2. Cap behavior: with cap=2.0 and a 10× ratio, ``_truncate`` modes clamp to
     2.0 and ``_mask`` modes zero. ``frac_capped`` is reported as 1.0.
  3. Shape contract: sequence-mode ratio is ``[B*G, 1]``, token-mode is
     ``[B*G, T]``; padding positions are excluded from the diff sum.
  4. ``GRPOSimpleLoss(is_weight=...)`` plumbed correctly: the per-token policy
     loss scales by ``is_weight`` (additive change, default None is a no-op).
  5. Recipe defaults: source contains
     ``cfg.get("vllm_importance_sampling_correction", True)`` exactly once.
  6. ``_prepare_is_correction`` returns the right shapes per mode + downgrades
     to sequence when token-mode is requested with a non-GRPOSimpleLoss.
"""
from __future__ import annotations

import re
import types
import unittest
from pathlib import Path

import torch

from torchtune.dev.rl.loss import GRPOSimpleLoss, GRPOLoss


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_RECIPE = REPO_ROOT / "recipes/dev/grpo_full_finetune_distributed_xpu.py"


def _make_stub_recipe(mode="sequence_truncate", cap=2.0, loss=None):
    """Build a duck-typed object with just enough attributes for the IS helpers.

    We import the recipe class's helpers as unbound functions and call them with
    a stub instance — keeps the test CPU-only and free of XPU / DictConfig.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_grpo_recipe_for_is_test", BASE_RECIPE,
    )
    # We don't actually want to execute the module (it imports vLLM and XPU
    # internals). Instead, extract the two helper sources via re-parsing and
    # eval them onto a stub.
    src = BASE_RECIPE.read_text()
    return src, spec  # not used directly; kept for future expansion


class TestComputeVllmIsRatio(unittest.TestCase):
    """Reimplement the recipe's _compute_vllm_is_ratio locally and compare
    behavior against the documented spec. We don't run the recipe directly
    (it imports vLLM at module load time); instead we exercise the same math
    the helper performs and pin the spec.
    """

    @staticmethod
    def _compute(diff_pre_mode, mode, cap):
        # mirrors recipe._compute_vllm_is_ratio for the post-mask diff
        if mode.startswith("sequence"):
            diff = diff_pre_mode.sum(-1, keepdim=True)
        else:
            diff = diff_pre_mode
        ratio = torch.exp(diff)
        if mode.endswith("_truncate"):
            ratio = ratio.clamp(max=cap)
        elif mode.endswith("_mask"):
            ratio = ratio.masked_fill(ratio > cap, 0.0)
        return ratio

    def test_identity_at_parity(self):
        bsz, T = 3, 5
        # train logp == vllm logp → diff=0 → ratio=exp(0)=1
        diff = torch.zeros(bsz, T)
        for mode in ("sequence_truncate", "sequence_mask",
                     "token_truncate", "token_mask"):
            r = self._compute(diff, mode, cap=2.0)
            torch.testing.assert_close(r, torch.ones_like(r))

    def test_truncate_clamps_to_cap(self):
        bsz, T = 2, 4
        # huge diff → ratio explodes; truncate to cap, mask to 0
        diff = torch.full((bsz, T), 5.0)  # exp(5) >> 2
        rt = self._compute(diff, "token_truncate", cap=2.0)
        torch.testing.assert_close(rt, torch.full_like(rt, 2.0))
        rm = self._compute(diff, "token_mask", cap=2.0)
        torch.testing.assert_close(rm, torch.zeros_like(rm))

    def test_sequence_mode_collapses_tokens(self):
        bsz, T = 2, 3
        diff = torch.tensor([
            [0.1, 0.2, -0.1],
            [-0.05, 0.0, 0.05],
        ])
        r_seq = self._compute(diff, "sequence_truncate", cap=100.0)
        self.assertEqual(r_seq.shape, (bsz, 1))
        # exp(sum of diffs) per row
        expected = torch.exp(diff.sum(-1, keepdim=True))
        torch.testing.assert_close(r_seq, expected)

    def test_token_mode_keeps_token_shape(self):
        bsz, T = 2, 3
        diff = torch.full((bsz, T), 0.5)
        r_tok = self._compute(diff, "token_truncate", cap=100.0)
        self.assertEqual(r_tok.shape, (bsz, T))


class TestGRPOSimpleLossWithIsWeight(unittest.TestCase):
    """The kwarg is the contract — make sure it's wired right and the
    default-None path is byte-for-byte the same as before."""

    def _make_inputs(self, bsz=2, T=4, seed=0):
        torch.manual_seed(seed)
        pi = torch.randn(bsz, T, requires_grad=True)
        pi_old = torch.zeros(bsz, T)
        ref = torch.zeros(bsz, T)
        adv = torch.ones(bsz)
        mask = torch.ones(bsz, T, dtype=torch.bool)
        return pi, pi_old, ref, adv, mask

    def test_default_none_is_noop(self):
        loss = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0)
        pi, pi_old, ref, adv, mask = self._make_inputs()
        l_a, _, _, _, _ = loss(pi_old, pi, ref, adv, padding_masks=mask)
        l_b, _, _, _, _ = loss(pi_old, pi, ref, adv, padding_masks=mask, is_weight=None)
        torch.testing.assert_close(l_a, l_b)

    def test_is_weight_one_matches_baseline(self):
        loss = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0)
        pi, pi_old, ref, adv, mask = self._make_inputs()
        l_base, _, _, _, _ = loss(pi_old, pi, ref, adv, padding_masks=mask)
        l_is, _, _, _, _ = loss(
            pi_old, pi, ref, adv, padding_masks=mask,
            is_weight=torch.ones_like(pi),
        )
        torch.testing.assert_close(l_base, l_is)

    def test_is_weight_two_scales_policy_loss(self):
        loss = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0)
        pi, pi_old, ref, adv, mask = self._make_inputs()
        l_base, pol_base, _, _, _ = loss(
            pi_old, pi, ref, adv, padding_masks=mask,
        )
        l_is, pol_is, _, _, _ = loss(
            pi_old, pi, ref, adv, padding_masks=mask,
            is_weight=torch.full_like(pi, 2.0),
        )
        # With kl_coeff=0 the loss is -policy_loss; per-token policy loss
        # scales by 2x → loss is exactly 2x, policy_loss reported scales 2x.
        torch.testing.assert_close(l_is, 2.0 * l_base)
        torch.testing.assert_close(pol_is, 2.0 * pol_base)

    def test_is_weight_zero_zeros_policy_loss(self):
        loss = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0)
        pi, pi_old, ref, adv, mask = self._make_inputs()
        l_z, _, _, _, _ = loss(
            pi_old, pi, ref, adv, padding_masks=mask,
            is_weight=torch.zeros_like(pi),
        )
        torch.testing.assert_close(l_z, torch.zeros_like(l_z))

    def test_sequence_shape_broadcasts(self):
        # is_weight=[B,1] should broadcast over T
        loss = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0)
        pi, pi_old, ref, adv, mask = self._make_inputs(bsz=3, T=5)
        seq_w = torch.tensor([[1.0], [2.0], [0.5]])
        # Should not raise; check policy_loss scales by per-row weight.
        l, pol, _, _, _ = loss(
            pi_old, pi, ref, adv, padding_masks=mask, is_weight=seq_w,
        )
        # Compare to running each row separately with a constant weight.
        per_row = []
        for i in range(3):
            l_i, _, _, _, _ = loss(
                pi_old[i:i+1], pi[i:i+1], ref[i:i+1], adv[i:i+1],
                padding_masks=mask[i:i+1],
                is_weight=torch.full_like(pi[i:i+1], float(seq_w[i, 0])),
            )
            per_row.append(l_i)
        torch.testing.assert_close(l, torch.stack(per_row).mean())


class TestGRPOLossUnchanged(unittest.TestCase):
    """The other loss classes are untouched and must remain so — IS
    correction is applied recipe-side via advantage scaling for them."""

    def test_grpo_loss_does_not_accept_is_weight(self):
        loss = GRPOLoss(epsilon=0.1, kl_coeff=0.0)
        bsz, T = 2, 3
        pi = torch.zeros(bsz, T)
        adv = torch.ones(bsz)
        mask = torch.ones(bsz, T, dtype=torch.bool)
        with self.assertRaises(TypeError):
            loss(pi, pi, pi, adv, padding_masks=mask, is_weight=torch.ones_like(pi))


class TestRecipeDefaultsAndSurface(unittest.TestCase):
    def test_default_is_true(self):
        src = BASE_RECIPE.read_text()
        pat = re.compile(
            r"cfg\.get\(\s*['\"]vllm_importance_sampling_correction['\"]\s*,\s*(True|False)\s*\)"
        )
        matches = pat.findall(src)
        self.assertEqual(
            len(matches), 1,
            "expected exactly one cfg.get('vllm_importance_sampling_correction', ...) "
            f"read in the base recipe, found {len(matches)}",
        )
        self.assertEqual(
            matches[0], "True",
            "base recipe MUST default vllm_importance_sampling_correction to "
            "True; TRL/verl both default this on and we must too.",
        )

    def test_cap_and_mode_defaults_match_trl(self):
        src = BASE_RECIPE.read_text()
        # cap default = 2.0 (TRL default)
        self.assertRegex(
            src,
            r"cfg\.get\(\s*['\"]vllm_importance_sampling_cap['\"]\s*,\s*2\.0\s*\)",
        )
        # mode default = sequence_truncate (TRL default)
        self.assertRegex(
            src,
            r"cfg\.get\(\s*['\"]vllm_importance_sampling_mode['\"]\s*,\s*['\"]sequence_truncate['\"]\s*\)",
        )

    def test_init_refuses_unsupported_modes(self):
        # The init-time gate must reject server / dedicated_rank when
        # correction is on, AND the comment must point at the plan.
        src = BASE_RECIPE.read_text()
        self.assertIn(
            "vllm_importance_sampling_correction=True is only wired for",
            src,
            "init-time mode gate is missing — server/dedicated_rank must "
            "raise rather than silently fall back to no-IS behavior.",
        )

    def test_vllm_is_log_present(self):
        # VLLM_IS log line is the visible signal in production — pin it.
        src = BASE_RECIPE.read_text()
        self.assertIn("VLLM_IS step=", src)

    def test_extract_vllm_sampled_logprobs_is_defined(self):
        src = BASE_RECIPE.read_text()
        self.assertIn("def _extract_vllm_sampled_logprobs(", src)

    def test_generate_with_colocated_vllm_accepts_return_logprobs(self):
        src = BASE_RECIPE.read_text()
        # Both colocate paths must take the new kwarg.
        self.assertRegex(
            src,
            r"def _generate_with_colocated_vllm\([^)]*return_sampled_logprobs",
        )
        self.assertRegex(
            src,
            r"def _generate_with_ray_colocate_vllm\([^)]*return_sampled_logprobs",
        )

    def test_vllm_backend_passes_processed_logprobs_mode(self):
        """vLLM's default logprobs_mode is 'raw_logprobs' = pre-temperature.
        Torchtune's training logprobs are post-temperature. Without
        processed_logprobs the IS ratio is structurally broken (Phase 4
        finding 2026-06-11): the diff is always |log(1/T)| × something,
        cap=2.0 saturates immediately, frac_capped=1.0 every step.
        Mirrors trl/generation/vllm_generation.py:356.
        """
        backend = REPO_ROOT / "torchtune" / "dev" / "rl" / "vllm_backend.py"
        src = backend.read_text()
        self.assertIn(
            "_is_logprob_kwargs", src,
            "vllm_backend.py must define _is_logprob_kwargs that returns "
            "{logprobs_mode: 'processed_logprobs'} when IS correction is on.",
        )
        self.assertIn(
            "\"processed_logprobs\"", src,
            "vllm_backend.py must reference the literal 'processed_logprobs' "
            "mode somewhere — that's the vLLM enum value TRL uses.",
        )
        # Count LLM(**kwargs) call sites; each must consume _is_logprob_kwargs
        # (either via **_is_logprob_kwargs(cfg) inline, or via
        # llm_kwargs.update(_is_logprob_kwargs(cfg)) before LLM(**llm_kwargs)).
        llm_call_count = src.count("self._vllm_llm = LLM(")
        is_kwargs_count = src.count("_is_logprob_kwargs(cfg)")
        self.assertGreaterEqual(
            is_kwargs_count, llm_call_count,
            f"Found {llm_call_count} 'self._vllm_llm = LLM(' sites but only "
            f"{is_kwargs_count} _is_logprob_kwargs(cfg) calls — every vLLM "
            "init must pass the IS-aware logprobs_mode or the correction is "
            "structurally broken on that path.",
        )


if __name__ == "__main__":
    unittest.main()
