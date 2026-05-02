# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU-safe tests for:
  1. _compute_maskfree_causal() — maskfree causal guard logic (Refactor 1)
  2. _log_varlen_status_once()  — one-shot varlen status log (attention_utils)

No XPU, no distributed init required. Runs in <1 s on CPU.
"""
import unittest
import unittest.mock as mock

import torch

import torchtune.modules.attention_utils as au
from torchtune.modules.attention_utils import _compute_maskfree_causal

PAD_ID = 0


def _make_qr(batch, prompt_len, resp_len, prompt_pads=None, resp_pads=None):
    """Build [B, P+R] integer token tensor. Non-PAD = 1."""
    total = prompt_len + resp_len
    tokens = torch.ones(batch, total, dtype=torch.long)
    for i in range(batch):
        if prompt_pads and prompt_pads[i] > 0:
            tokens[i, prompt_len - prompt_pads[i]: prompt_len] = PAD_ID
        if resp_pads and resp_pads[i] > 0:
            tokens[i, total - resp_pads[i]:] = PAD_ID
    return tokens


# ---------------------------------------------------------------------------
# 1. TestMaskfreeGuard — unit tests for _compute_maskfree_causal()
# ---------------------------------------------------------------------------

class TestMaskfreeGuard(unittest.TestCase):

    def _call(self, env_set, device_type, packing_enabled, qr, ctx_len):
        return _compute_maskfree_causal(
            env_set=env_set,
            device_type=device_type,
            packing_enabled=packing_enabled,
            query_responses=qr,
            context_length=ctx_len,
            pad_id=PAD_ID,
        )

    def test_env_not_set(self):
        qr = _make_qr(2, 8, 4)
        result = self._call(False, "xpu", False, qr, 8)
        self.assertEqual(result, (False, "env not set"))

    def test_wrong_device(self):
        qr = _make_qr(2, 8, 4)
        result = self._call(True, "cpu", False, qr, 8)
        self.assertEqual(result, (False, "device != xpu"))

    def test_packing_enabled(self):
        qr = _make_qr(2, 8, 4)
        result = self._call(True, "xpu", True, qr, 8)
        self.assertEqual(result, (False, "packing enabled"))

    def test_prompt_padding_fallback(self):
        # Prompt region ([:, :8]) has a PAD token
        qr = _make_qr(2, 8, 4, prompt_pads=[2, 0])
        result = self._call(True, "xpu", False, qr, 8)
        self.assertEqual(result, (False, "prompt padding detected"))

    def test_clean_batch_engages(self):
        qr = _make_qr(2, 8, 4, prompt_pads=[0, 0], resp_pads=[0, 0])
        result = self._call(True, "xpu", False, qr, 8)
        self.assertEqual(result, (True, None))

    def test_response_only_padding_ok(self):
        # PAD tokens only in the response region — must NOT trigger the guard
        qr = _make_qr(2, 8, 4, prompt_pads=[0, 0], resp_pads=[3, 1])
        result = self._call(True, "xpu", False, qr, 8)
        self.assertEqual(result, (True, None))


# ---------------------------------------------------------------------------
# 2. TestVarlenLogStatus — unit tests for _log_varlen_status_once()
# ---------------------------------------------------------------------------

class TestVarlenLogStatus(unittest.TestCase):

    def setUp(self):
        au._reset_varlen_log_for_testing()

    def _log(self, mask, is_causal, dropout_p, device_type):
        au._log_varlen_status_once(mask, is_causal, dropout_p, device_type)

    def test_disabled_env_unset(self):
        """When _USE_IPEX_VARLEN is False, log 'varlen=disabled'."""
        with mock.patch.object(au, '_USE_IPEX_VARLEN', False), \
             self.assertLogs(level='INFO') as cm:
            self._log(None, True, 0.0, "cpu")
        self.assertTrue(
            any("varlen=disabled" in m for m in cm.output),
            f"Expected 'varlen=disabled' in logs, got: {cm.output}"
        )

    def test_skipped_device_cpu(self):
        """device=cpu triggers 'requested-but-skipped' with 'device=cpu'."""
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm:
            self._log(None, True, 0.0, "cpu")
        combined = " ".join(cm.output)
        self.assertIn("varlen=requested-but-skipped", combined)
        self.assertIn("device=cpu", combined)

    def test_skipped_mask_not_none(self):
        """Non-None mask triggers 'requested-but-skipped' with 'mask is not None'."""
        dummy_mask = torch.ones(1, 4, 4, dtype=torch.bool)
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm:
            self._log(dummy_mask, True, 0.0, "cpu")
        self.assertIn("mask is not None", " ".join(cm.output))

    def test_skipped_not_causal(self):
        """is_causal=False triggers 'requested-but-skipped' with 'is_causal=False'."""
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm:
            self._log(None, False, 0.0, "cpu")
        self.assertIn("is_causal=False", " ".join(cm.output))

    def test_skipped_dropout(self):
        """Non-zero dropout triggers 'requested-but-skipped' with 'dropout_p=0.1'."""
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm:
            self._log(None, True, 0.1, "cpu")
        self.assertIn("dropout_p=0.1", " ".join(cm.output))

    def test_skipped_multi_reason(self):
        """Multiple skip conditions: both reasons appear in the log message."""
        dummy_mask = torch.ones(1, 4, 4, dtype=torch.bool)
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm:
            # mask + is_causal=False, no dropout, device=cpu
            self._log(dummy_mask, False, 0.0, "cpu")
        combined = " ".join(cm.output)
        self.assertIn("mask is not None", combined)
        self.assertIn("is_causal=False", combined)

    def test_one_shot_fires_once(self):
        """Two calls with the same args produce exactly 1 log record."""
        with mock.patch.object(au, '_USE_IPEX_VARLEN', False), \
             self.assertLogs(level='INFO') as cm:
            self._log(None, True, 0.0, "cpu")
            self._log(None, True, 0.0, "cpu")  # no-op: _VARLEN_LOG_DONE=True
        self.assertEqual(len(cm.records), 1, f"Expected 1 log, got {len(cm.records)}")

    def test_engaged_xpu_mock_no_grad(self):
        """All conditions met on 'xpu' in no-grad mode → 'varlen=engaged'."""
        import torch
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm, \
             torch.no_grad():
            self._log(None, True, 0.0, "xpu")
        self.assertIn("varlen=engaged", " ".join(cm.output))

    def test_no_grad_only_xpu_mock_with_grad(self):
        """All conditions met on 'xpu' but grad enabled → 'varlen=no-grad-only' (training fwd guard)."""
        import torch
        assert torch.is_grad_enabled(), "grad must be enabled for this test"
        with mock.patch.object(au, '_USE_IPEX_VARLEN', True), \
             mock.patch.object(au, '_ipex_varlen_attention', object()), \
             self.assertLogs(level='INFO') as cm:
            self._log(None, True, 0.0, "xpu")
        self.assertIn("varlen=no-grad-only", " ".join(cm.output))


if __name__ == "__main__":
    unittest.main()
