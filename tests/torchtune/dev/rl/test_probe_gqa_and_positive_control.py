# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Guards for the split-SDPA flash-eligibility probe.

Job 8830040 wasted a debug-queue slot and printed ``VERDICT: FAIL`` for a feature that
may well be viable, because the probe called
``F.scaled_dot_product_attention`` on GQA shapes (32 query heads / 8 KV heads) without
``enable_gqa=True``. Under ``sdpa_kernel([FLASH_ATTENTION])`` that surfaces as
``RuntimeError: No available kernel`` -- textually identical to a real shape rejection --
so every case "failed", *including the configuration production runs successfully*.

Two defects, two guards:

1. **GQA dispatch.** Every ``F.scaled_dot_product_attention`` call in the probe must pass
   ``enable_gqa=True``, since ``HEADS_Q != HEADS_KV``.
2. **Missing positive control.** The probe had a negative control (a case that must be
   rejected) but no known-good baseline that must PASS. A negative control only shows the
   probe can say "no"; it cannot distinguish a discriminating probe from one that rejects
   everything -- which is exactly what happened, and the negative control passed vacuously.

Also pins the math: the split + LSE merge must reproduce bottom-right causal attention.

See ``memory/feedback_probe_needs_known_good_baseline_not_just_reject_control_20260916.md``.
"""
import ast
from pathlib import Path

import pytest
import torch

PROBE = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "bioreason"
    / "probe_split_sdpa_flash_eligibility.py"
)


@pytest.fixture(scope="module")
def probe_tree():
    if not PROBE.exists():
        pytest.skip(f"probe not present at {PROBE}")
    return ast.parse(PROBE.read_text()), PROBE.read_text()


def _sdpa_calls(tree):
    """Every ``F.scaled_dot_product_attention(...)`` call node in the probe."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "scaled_dot_product_attention":
                out.append(node)
    return out


class TestProbeGqaDispatch:
    def test_probe_is_actually_gqa(self, probe_tree):
        _, src = probe_tree
        assert "HEADS_Q = 32" in src and "HEADS_KV = 8" in src, (
            "if the probe stops being GQA, the enable_gqa guard below is moot -- "
            "update both together"
        )

    def test_every_functional_sdpa_call_enables_gqa(self, probe_tree):
        tree, _ = probe_tree
        calls = _sdpa_calls(tree)
        assert calls, "expected at least one F.scaled_dot_product_attention call"
        for call in calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "enable_gqa" in kwargs, (
                f"F.scaled_dot_product_attention at line {call.lineno} omits "
                "enable_gqa=True; on 32q/8kv this reports 'No available kernel' "
                "inside sdpa_kernel([FLASH_ATTENTION]) and fakes a shape rejection"
            )
            for kw in call.keywords:
                if kw.arg == "enable_gqa":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is True


class TestProbeHasPositiveControl:
    def test_has_a_must_pass_baseline(self, probe_tree):
        _, src = probe_tree
        assert "MUST PASS" in src, (
            "the probe needs a known-good positive baseline (the production "
            "TORCHTUNE_USE_XPU_FLASH config: square, is_causal=True) asserted MUST "
            "PASS. Without it a probe that rejects everything looks like a feature "
            "verdict -- see job 8830040."
        )

    def test_still_has_a_negative_control(self, probe_tree):
        _, src = probe_tree
        assert "must be REJECTED" in src, "negative control must not be dropped"

    def test_baseline_is_checked_before_the_experimental_cases(self, probe_tree):
        _, src = probe_tree
        assert src.index("MUST PASS") < src.index("cross  resp->prefix"), (
            "assert the known-good baseline first; a failure there means the probe "
            "is broken and nothing after it is evidence about the feature"
        )


GQA_FOLD_PROBE = PROBE.parent / "probe_split_sdpa_gqa_fold.py"


@pytest.fixture(scope="module")
def gqa_fold_probe():
    if not GQA_FOLD_PROBE.exists():
        pytest.skip(f"probe not present at {GQA_FOLD_PROBE}")
    src = GQA_FOLD_PROBE.read_text()
    return ast.parse(src), src


class TestGqaFoldProbeGateIsolation:
    """Jobs 8830248/8830291/8830313: three debug runs, one crash discovered per run.

    The gate #3b probe wrapped its whole ladder in a single ``try``. Gate 3's negative
    control threw on a ``.view()``, and gate 4 -- which asks an independent question
    (does the prefix KV stay ``B`` rows?) and shares no code with gate 3 -- never ran on
    any of the three allocations. See
    ``memory/feedback_probe_gates_need_independent_exception_boundaries_20260916.md``.
    """

    def test_gate_4_has_its_own_exception_boundary(self, gqa_fold_probe):
        tree, _ = gqa_fold_probe
        main = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        guarded = {
            node.func.id
            for handler_parent in ast.walk(main)
            if isinstance(handler_parent, ast.Try)
            for node in ast.walk(handler_parent)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("gate_")
            # the call must be inside a Try whose body is NOT the whole ladder: a Try
            # containing more than one distinct gate call is the coupling we are banning
            and len(
                {
                    c.func.id
                    for c in ast.walk(handler_parent)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id.startswith("gate_")
                }
            )
            == 1
        }
        assert "gate_4_kv_memory_is_b_rows" in guarded, (
            "gate 4 asks an independent question and must not be able to be skipped by "
            "a crash in gate 3; give it its own try/except"
        )

    def test_crashed_gate_is_reported_as_a_failure(self, gqa_fold_probe):
        """A bare ``traceback.print_exc()`` leaves the verdict silent about the gate."""
        tree, _ = gqa_fold_probe
        main = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        inner = [
            h
            for t in ast.walk(main)
            if isinstance(t, ast.Try)
            for h in t.handlers
            if any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id.startswith("gate_")
                for c in ast.walk(t)
            )
        ]
        reporting = [
            h
            for h in inner
            if any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "_report"
                for c in ast.walk(h)
            )
        ]
        assert reporting, (
            "at least one per-gate handler must call _report(False, ...) so a crash "
            "shows up as a named FAILED gate rather than as an absent line"
        )

    def test_no_bare_view_calls_on_folded_tensors(self, gqa_fold_probe):
        """``view`` cannot fold non-adjacent axes of a BSHD tensor; ``reshape`` can.

        Two of the three wasted runs were a ``.view()`` raising "size is not compatible
        with input tensor's size and stride". The two surviving calls are the ones that
        demonstrably executed on HW (job 8830313); any NEW one is almost certainly the
        same bug, so this pins the count rather than banning the construct outright.
        """
        tree, _ = gqa_fold_probe
        views = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "view"
        ]
        assert len(views) <= 2, (
            f"found {len(views)} .view() calls; only the 2 HW-validated ones are "
            "expected. Prefer .reshape() in probe code -- see jobs 8830291/8830313"
        )


class TestSplitMergeMath:
    """The decomposition itself is sound -- only the dispatch was wrong."""

    def test_split_plus_lse_merge_equals_bottom_right_causal(self):
        torch.manual_seed(0)
        hq, hkv, d, prefix, resp = 8, 2, 16, 64, 24

        def mk(h, n):
            return torch.randn(1, h, n, d, dtype=torch.float64)

        q, k_p, v_p, k_r, v_r = (
            mk(hq, resp),
            mk(hkv, prefix),
            mk(hkv, prefix),
            mk(hkv, resp),
            mk(hkv, resp),
        )

        def attn_lse(q, k, v, causal):
            kk = k.repeat_interleave(hq // hkv, dim=1)
            vv = v.repeat_interleave(hq // hkv, dim=1)
            s = q @ kk.transpose(-1, -2) / d**0.5
            if causal:
                n_q, n_k = s.shape[-2], s.shape[-1]
                i = torch.arange(n_q).unsqueeze(-1)
                j = torch.arange(n_k).unsqueeze(0)
                # bottom-right alignment: query i sits at absolute position
                # (n_k - n_q) + i. is_causal=True would be top-left, a different fn.
                s = s.masked_fill(j > i + (n_k - n_q), float("-inf"))
            return torch.softmax(s, dim=-1) @ vv, torch.logsumexp(s, dim=-1)

        o_c, l_c = attn_lse(q, k_p, v_p, False)  # cross: resp -> prefix, not causal
        o_s, l_s = attn_lse(q, k_r, v_r, True)  # self:  resp -> resp, causal

        la, lb = l_c.unsqueeze(-1), l_s.unsqueeze(-1)
        m = torch.maximum(la, lb)
        wa, wb = torch.exp(la - m), torch.exp(lb - m)
        split = (o_c * wa + o_s * wb) / (wa + wb)

        ref, _ = attn_lse(
            q, torch.cat([k_p, k_r], 2), torch.cat([v_p, v_r], 2), True
        )
        assert split.shape == ref.shape
        assert (split - ref).abs().max().item() < 1e-12

    def test_cross_block_must_not_be_causal(self):
        """Sanity: making the cross block causal changes the answer, so the
        is_causal=False choice is load-bearing, not cosmetic."""
        torch.manual_seed(0)
        hq, d, prefix, resp = 4, 8, 32, 12
        q = torch.randn(1, hq, resp, d, dtype=torch.float64)
        k = torch.randn(1, hq, prefix, d, dtype=torch.float64)
        v = torch.randn(1, hq, prefix, d, dtype=torch.float64)
        s = q @ k.transpose(-1, -2) / d**0.5
        free = torch.softmax(s, dim=-1) @ v
        i = torch.arange(resp).unsqueeze(-1)
        j = torch.arange(prefix).unsqueeze(0)
        causal = torch.softmax(
            s.masked_fill(j > i + (prefix - resp), float("-inf")), dim=-1
        ) @ v
        assert (free - causal).abs().max().item() > 1e-6
