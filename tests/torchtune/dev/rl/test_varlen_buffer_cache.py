# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU-safe tests for the _ipex_varlen_call persistent output-buffer cache contract
inside _sdpa_or_flex_attention().

Three invariants:
  - No-grad, same shape twice → torch.empty_like called exactly once (cache hit).
  - Grad-enabled, same shape twice → torch.empty_like called at least twice (fresh alloc).
  - No-grad, shape change → torch.empty_like called twice (cache miss on new shape).

No XPU required. IPEX kernel is replaced by a CPU-runnable noop.
Runs in <0.5 s on CPU.
"""
import unittest
import unittest.mock as mock

import torch

import torchtune.modules.attention_utils as au


def _noop_varlen(*args, **kwargs):
    """CPU-runnable stand-in for _ipex_varlen_attention. Writes zeros to out (arg[3])."""
    out = args[3]
    out.zero_()


def _get_ipex_varlen_call(attn_fn):
    """
    Extract _ipex_varlen_call from the closure of the returned attention function.

    _sdpa_or_flex_attention() returns either:
      - _sdpa_call (when _SUPPORTS_FLEX_ATTENTION is False)
      - _attention_call (which wraps _sdpa_call when _SUPPORTS_FLEX_ATTENTION is True)

    _ipex_varlen_call is a free variable of _sdpa_call in both cases.
    """
    # Direct case: attn_fn IS _sdpa_call
    if '_ipex_varlen_call' in attn_fn.__code__.co_freevars:
        idx = attn_fn.__code__.co_freevars.index('_ipex_varlen_call')
        return attn_fn.__closure__[idx].cell_contents

    # Indirect case: attn_fn is _attention_call; _sdpa_call is in its closure
    for cell in (attn_fn.__closure__ or []):
        try:
            obj = cell.cell_contents
            if (callable(obj) and hasattr(obj, '__code__')
                    and '_ipex_varlen_call' in obj.__code__.co_freevars):
                idx = obj.__code__.co_freevars.index('_ipex_varlen_call')
                return obj.__closure__[idx].cell_contents
        except ValueError:
            pass

    raise RuntimeError(
        "Could not find _ipex_varlen_call in closure hierarchy. "
        "The internal structure of _sdpa_or_flex_attention() may have changed."
    )


class TestVarlenBufferCache(unittest.TestCase):
    """
    Tests the _ipex_varlen_call buffer-reuse contract.

    Each test creates a FRESH closure (fresh cache dicts) so tests don't share state.
    We extract _ipex_varlen_call directly, bypassing the device check in _sdpa_call,
    and call it with CPU tensors + the _noop_varlen stand-in.
    """

    def _make_qkv(self, b, h, s, d):
        return [torch.randn(b, h, s, d) for _ in range(3)]

    def _get_varlen_call(self):
        """Fresh closure — isolates cache dicts from other tests."""
        attn_fn = au._sdpa_or_flex_attention()
        return _get_ipex_varlen_call(attn_fn)

    def test_nograd_same_shape_reuses_buffer(self):
        """No-grad + same shape twice → empty_like called once (second call hits cache)."""
        varlen_call = self._get_varlen_call()
        q, k, v = self._make_qkv(2, 4, 8, 16)
        with mock.patch.object(au, '_ipex_varlen_attention', _noop_varlen):
            with torch.no_grad(), \
                 mock.patch('torch.empty_like', wraps=torch.empty_like) as m:
                varlen_call(q, k, v)
                varlen_call(q, k, v)
        self.assertEqual(
            m.call_count, 1,
            f"No-grad same shape should reuse cached buffer (got {m.call_count} empty_like calls)"
        )

    def test_grad_enabled_always_allocates_fresh(self):
        """Grad-enabled + same shape twice → empty_like called at least twice (no cache)."""
        varlen_call = self._get_varlen_call()
        q, k, v = self._make_qkv(2, 4, 8, 16)
        with mock.patch.object(au, '_ipex_varlen_attention', _noop_varlen):
            with mock.patch('torch.empty_like', wraps=torch.empty_like) as m:
                varlen_call(q, k, v)
                varlen_call(q, k, v)
        self.assertGreaterEqual(
            m.call_count, 2,
            f"Grad-enabled should allocate fresh each call (got {m.call_count} empty_like calls)"
        )

    def test_nograd_shape_change_reallocates(self):
        """No-grad + shape change → empty_like called twice (second call = cache miss)."""
        varlen_call = self._get_varlen_call()
        q1, k1, v1 = self._make_qkv(2, 4, 8, 16)
        q2, k2, v2 = self._make_qkv(2, 4, 12, 16)  # different seq len → new cache key
        with mock.patch.object(au, '_ipex_varlen_attention', _noop_varlen):
            with torch.no_grad(), \
                 mock.patch('torch.empty_like', wraps=torch.empty_like) as m:
                varlen_call(q1, k1, v1)
                varlen_call(q2, k2, v2)
        self.assertEqual(
            m.call_count, 2,
            f"No-grad shape change should reallocate (got {m.call_count} empty_like calls)"
        )

    def _get_caches(self, varlen_call):
        """Pull the three cache dicts out of _ipex_varlen_call's closure."""
        names = ("_varlen_out_cache", "_varlen_alibi_cache", "_varlen_seqlens_cache")
        out = {}
        fv = varlen_call.__code__.co_freevars
        for n in names:
            self.assertIn(n, fv, f"{n} not a free var of _ipex_varlen_call")
            out[n] = varlen_call.__closure__[fv.index(n)].cell_contents
        return out

    def test_nograd_cache_is_bounded(self):
        """
        THE regression test for the no-FSDP colocate creep root cause.

        Variable-seqlen RL produces a new (b,h,s,d) key nearly every step. The old
        unbounded dict cached one ~14.7 MiB output buffer per distinct shape FOREVER
        (~0.44 GiB/step live-memory leak in the ref forward). The bounded cache must
        cap the number of retained entries regardless of how many distinct shapes
        are seen — while still reusing within a step.
        """
        varlen_call = self._get_varlen_call()
        caches = self._get_caches(varlen_call)
        # cap is read from env at closure-creation; default 8. Feed many MORE
        # distinct seqlens than the cap and assert the cache never exceeds it.
        cap = au_cache_max = None
        # derive the active cap the same way the module does
        import os as _os
        cap = max(1, int(_os.environ.get("TORCHTUNE_VARLEN_CACHE_MAX", "8")))
        n_shapes = cap + 20
        with mock.patch.object(au, '_ipex_varlen_attention', _noop_varlen):
            with torch.no_grad():
                for i in range(n_shapes):
                    s = 8 + i  # distinct seq len each iteration → distinct cache key
                    q, k, v = self._make_qkv(2, 4, s, 16)
                    varlen_call(q, k, v)
        for name, c in caches.items():
            self.assertLessEqual(
                len(c), cap,
                f"{name} grew to {len(c)} entries with cap={cap} — cache is UNBOUNDED "
                f"(the variable-seqlen RL leak is back)."
            )
        # all three caches must stay key-aligned (evicted together)
        self.assertEqual(
            set(caches["_varlen_out_cache"].keys()),
            set(caches["_varlen_alibi_cache"].keys()),
            "out/alibi caches drifted — eviction is not joint",
        )

    def test_nograd_repeated_same_shape_stays_single_entry(self):
        """Same shape many times → exactly ONE cached entry (within-step reuse intact)."""
        varlen_call = self._get_varlen_call()
        caches = self._get_caches(varlen_call)
        q, k, v = self._make_qkv(2, 4, 8, 16)
        with mock.patch.object(au, '_ipex_varlen_attention', _noop_varlen):
            with torch.no_grad():
                for _ in range(50):
                    varlen_call(q, k, v)
        self.assertEqual(
            len(caches["_varlen_out_cache"]), 1,
            "Repeated identical shape must collapse to one cache entry (the speedup).",
        )


if __name__ == "__main__":
    unittest.main()
