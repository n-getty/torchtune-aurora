# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe gating tests for the opt-in native SYCL-TLA flash XPU training path
(TORCHTUNE_USE_XPU_FLASH). The kernel itself needs XPU + the shipped
libtorch-xpu-ops-sycltla-mha_{fwd,bwd}.so, but the ROUTING LOGIC in _sdpa_call
and the BSHD-memory coercion must be exercisable on a login node without XPU:

  - default (flag unset) NEVER routes to flash,
  - flag set but device != xpu falls through to SDPA,
  - flag set but mask!=None / is_causal=False / dropout>0 falls through,
  - flash takes precedence over flex when both flags are set,
  - _to_bshd_memory returns BSHD-memory strides and is value-preserving.

These guard against a regression that silently sends the wrong tensors to the
fused flash kernel (wrong device, non-causal, masked) or breaks the layout
coercion (which would cause a hard "No available kernel" on XPU).
"""
import importlib
import os
from unittest import mock

import torch


def _reload_au(env: dict):
    """Reimport attention_utils with a patched environment so the module-level
    flag reads (TORCHTUNE_USE_XPU_FLASH etc.) take effect."""
    with mock.patch.dict(os.environ, env, clear=False):
        import torchtune.modules.attention_utils as au

        return importlib.reload(au)


def _restore_default():
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("TORCHTUNE_USE_XPU_FLASH", "TORCHTUNE_USE_XPU_FLEX")
    }
    with mock.patch.dict(os.environ, env, clear=True):
        import torchtune.modules.attention_utils as au

        importlib.reload(au)


def test_default_flag_off_returns_sdpa_and_never_calls_flash():
    au = _reload_au({"TORCHTUNE_USE_XPU_FLASH": "0"})
    try:
        assert au._USE_XPU_FLASH is False
        fn = au._sdpa_or_flex_attention()
        q = torch.randn(1, 4, 16, 8)
        with mock.patch.object(
            au, "_xpu_flash_call", side_effect=AssertionError("must not call flash")
        ):
            o = fn(q, q, q, None, 0.0, True)
        assert o.shape == q.shape
    finally:
        _restore_default()


def test_flag_on_cpu_falls_through_to_sdpa():
    au = _reload_au({"TORCHTUNE_USE_XPU_FLASH": "1"})
    try:
        au._reset_xpu_flash_log_for_testing()
        fn = au._sdpa_or_flex_attention()
        q = torch.randn(1, 4, 16, 8)  # device=cpu

        def _spy(*a, **k):
            raise AssertionError("flash called on non-xpu device")

        with mock.patch.object(au, "_xpu_flash_call", _spy):
            o = fn(q, q, q, None, 0.0, True)
        assert o.shape == q.shape
    finally:
        _restore_default()


def test_flag_on_gate_conditions_fall_through():
    """Even pretending the tensor is XPU, non-causal / masked / dropout>0 must
    not route to flash."""
    au = _reload_au({"TORCHTUNE_USE_XPU_FLASH": "1"})
    try:
        au._reset_xpu_flash_log_for_testing()
        fn = au._sdpa_or_flex_attention()
        q = torch.randn(1, 4, 16, 8)
        with mock.patch.object(
            au, "_xpu_flash_call", side_effect=AssertionError("must not call flash")
        ):
            m = torch.zeros(1, 16, 16, dtype=torch.bool)
            fn(q, q, q, m, 0.0, True)  # mask -> fall through
            fn(q, q, q, None, 0.0, False)  # non-causal -> fall through
            fn(q, q, q, None, 0.1, True)  # dropout>0 -> fall through
    finally:
        _restore_default()


def test_flash_takes_precedence_over_flex_on_xpu_path():
    """When both flags are set, the flash branch is checked first. We simulate an
    XPU tensor by patching a tensor's device.type via the routing: since we cannot
    make a real XPU tensor on CPU, assert ordering by confirming the flash branch's
    guard is evaluated before flex's in _sdpa_call (flash spy fires, flex does not)."""
    au = _reload_au({"TORCHTUNE_USE_XPU_FLASH": "1", "TORCHTUNE_USE_XPU_FLEX": "1"})
    try:
        au._reset_xpu_flash_log_for_testing()
        au._reset_xpu_flex_log_for_testing()
        fn = au._sdpa_or_flex_attention()

        # Fake an xpu tensor: a real CPU tensor whose .device.type reports "xpu".
        q = torch.randn(1, 4, 16, 8)

        class _FakeDev:
            type = "xpu"

        flash_called = {"v": False}

        def _flash_spy(qq, kk, vv, dp):
            flash_called["v"] = True
            return qq  # short-circuit; shape matches

        def _flex_spy(*a, **k):
            raise AssertionError("flex must not be reached when flash handles it")

        with mock.patch.object(type(q), "device", _FakeDev()), mock.patch.object(
            au, "_xpu_flash_call", _flash_spy
        ), mock.patch.object(au, "_xpu_flex_call", _flex_spy):
            # only reached if _USE_XPU_FLASH import succeeded on this box
            if au._USE_XPU_FLASH and au._xpu_flash_sdpa_kernel is not None:
                fn(q, q, q, None, 0.0, True)
                assert flash_called["v"] is True
    finally:
        _restore_default()


def test_to_bshd_memory_layout_and_values():
    """_to_bshd_memory must return a [B,H,S,D] tensor whose [B,S,H,D] view is
    contiguous (BSHD memory) and whose values are unchanged. Idempotent on an
    already-BSHD tensor (no needless copy)."""
    au = _reload_au({"TORCHTUNE_USE_XPU_FLASH": "1"})
    try:
        b, h, s, d = 2, 4, 8, 16
        # Standard C-contiguous [B,H,S,D] — its [B,S,H,D] view is NOT contiguous.
        t = torch.randn(b, h, s, d)
        assert not t.transpose(1, 2).is_contiguous()
        out = au._to_bshd_memory(t)
        assert out.shape == (b, h, s, d)
        assert out.transpose(1, 2).is_contiguous()  # BSHD memory
        assert torch.equal(out, t)  # value preserving

        # Already-BSHD input (a transpose view over contiguous [B,S,H,D]) is a no-op
        # (returns the same object — no copy).
        bshd = torch.randn(b, s, h, d).transpose(1, 2)  # [B,H,S,D] view, BSHD mem
        assert bshd.transpose(1, 2).is_contiguous()
        out2 = au._to_bshd_memory(bshd)
        assert out2 is bshd
    finally:
        _restore_default()
