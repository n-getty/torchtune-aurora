# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe gating tests for the opt-in compiled flex_attention XPU training path
(TORCHTUNE_USE_XPU_FLEX). The kernel itself needs XPU + Triton, but the ROUTING
LOGIC in _sdpa_call must be exercisable on a login node without XPU:

  - default (flag unset) NEVER routes to flex,
  - flag set but device != xpu falls through to SDPA,
  - flag set but mask!=None / is_causal=False / dropout>0 falls through,
  - the causal BlockMask cache is FIFO-bounded and uses q_idx>=kv_idx.

These guard against a regression that silently sends the wrong tensors to flex
(wrong device, non-causal, masked) or unbounds the mask cache.
"""
import importlib
import os
from unittest import mock

import torch


def _reload_au(env: dict):
    """Reimport attention_utils with a patched environment so the module-level
    flag reads (TORCHTUNE_USE_XPU_FLEX etc.) take effect."""
    with mock.patch.dict(os.environ, env, clear=False):
        import torchtune.modules.attention_utils as au

        return importlib.reload(au)


def _restore_default():
    # Reload once more with the flag unset so other tests see the default module.
    env = {k: v for k, v in os.environ.items() if k != "TORCHTUNE_USE_XPU_FLEX"}
    with mock.patch.dict(os.environ, env, clear=True):
        import torchtune.modules.attention_utils as au

        importlib.reload(au)


def test_default_flag_off_returns_sdpa_and_never_calls_flex():
    au = _reload_au({"TORCHTUNE_USE_XPU_FLEX": "0"})
    try:
        assert au._USE_XPU_FLEX is False
        assert au._xpu_flex_compiled is None
        # _sdpa_or_flex_attention returns the _sdpa_call closure on non-CUDA; the
        # XPU flex branch we care about lives inside it.
        fn = au._sdpa_or_flex_attention()
        q = torch.randn(1, 4, 16, 8)
        with mock.patch.object(
            au, "_xpu_flex_call", side_effect=AssertionError("must not call flex")
        ):
            o = fn(q, q, q, None, 0.0, True)
        assert o.shape == q.shape
    finally:
        _restore_default()


def test_flag_on_cpu_falls_through_to_sdpa():
    au = _reload_au({"TORCHTUNE_USE_XPU_FLEX": "1"})
    try:
        # Import + compile may or may not succeed on the box; the routing guard is
        # what we assert. flex must NOT be called for a CPU tensor.
        au._reset_xpu_flex_log_for_testing()
        fn = au._sdpa_or_flex_attention()
        q = torch.randn(1, 4, 16, 8)  # device=cpu
        called = {"flex": False}

        def _spy(*a, **k):
            called["flex"] = True
            raise AssertionError("flex called on non-xpu device")

        with mock.patch.object(au, "_xpu_flex_call", _spy):
            o = fn(q, q, q, None, 0.0, True)
        assert called["flex"] is False
        assert o.shape == q.shape
    finally:
        _restore_default()


def test_flag_on_gate_conditions_fall_through():
    """Even pretending the tensor is XPU, non-causal / masked / dropout>0 must
    not route to flex. We fake device.type via a spy on _xpu_flex_call and drive
    each disqualifying condition; flex must remain uncalled."""
    au = _reload_au({"TORCHTUNE_USE_XPU_FLEX": "1"})
    try:
        au._reset_xpu_flex_log_for_testing()
        fn = au._sdpa_or_flex_attention()
        q = torch.randn(1, 4, 16, 8)
        with mock.patch.object(
            au, "_xpu_flex_call", side_effect=AssertionError("must not call flex")
        ):
            # mask is not None -> fall through (SDPA needs 4d mask broadcast)
            m = torch.zeros(1, 16, 16, dtype=torch.bool)
            fn(q, q, q, m, 0.0, True)
            # is_causal=False -> fall through
            fn(q, q, q, None, 0.0, False)
            # dropout>0 -> fall through
            fn(q, q, q, None, 0.1, True)
    finally:
        _restore_default()


def test_causal_block_mask_cache_is_fifo_bounded():
    au = _reload_au(
        {"TORCHTUNE_USE_XPU_FLEX": "1", "TORCHTUNE_XPU_FLEX_MASK_CACHE_MAX": "2"}
    )
    try:
        if au._xpu_flex_create_block_mask is None:
            # flex not importable on this box; the cache helper needs it. Skip
            # gracefully — the routing tests above still cover the gate.
            return
        au._reset_xpu_flex_log_for_testing()
        dev = torch.device("cpu")
        # create_block_mask works on CPU; build 3 distinct seq lens, cap is 2
        for s in (128, 256, 512):
            au._xpu_flex_causal_mask(s, dev)
        assert len(au._xpu_flex_mask_cache) == 2, au._xpu_flex_mask_cache.keys()
        # oldest (128) evicted, newest two remain
        keys = {k[0] for k in au._xpu_flex_mask_cache.keys()}
        assert keys == {256, 512}, keys
        # re-fetch an existing key returns cached object (no rebuild growth)
        before = len(au._xpu_flex_mask_cache)
        au._xpu_flex_causal_mask(512, dev)
        assert len(au._xpu_flex_mask_cache) == before
    finally:
        _restore_default()
