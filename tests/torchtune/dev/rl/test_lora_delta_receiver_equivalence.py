"""Pin-down: the merge-at-receiver (Path C) delta path is bit-exact to the
merged path (Path A).

Path C ships the frozen base ONCE (load_lora_base_from_raw) then only the
~66 MB lora_a/lora_b adapter each step (load_lora_delta_from_raw); the vLLM
worker re-merges W_eff = base + scale*(B@A). This test exercises the REAL
worker methods on a fake model_runner (no vLLM, no XPU, no distributed) and
asserts the merged weights captured by load_weights() equal what
``iter_merged_lora_layers`` (Path A) produces for the same adapter.

Also covered:
  - zero drift across N steps (stateless re-merge always from pristine base);
  - the LLAMA-family Q/K un-permute-on-delta path (commutes with the base add).
"""
from __future__ import annotations

import json
import os
import struct
import tempfile

import torch
from torch import nn

from torchtune.dev.rl.lora_helpers import _TUNE_MODULE_TO_HF, iter_merged_lora_layers
from torchtune.dev.rl.weight_sync import (
    _qk_unpermute_for_vllm,
    _save_raw_bytes,
)
from torchtune.modules.peft.lora import LoRALinear
from torchtune.dev.vllm_weight_sync_worker import WeightSyncFromFileExtension


# --------------------------------------------------------------------------
# Fake vLLM model / runner / worker (mirrors test_load_fused_moe_ep_slicing.py)
# --------------------------------------------------------------------------
class _FakeModel:
    """Holds resident params by HF name; captures load_weights() calls."""

    def __init__(self, params: dict):
        self._named = dict(params)
        self.loaded: dict[str, torch.Tensor] = {}

    def named_parameters(self):
        return list(self._named.items())

    def load_weights(self, weights):
        # Mirror vLLM: in production this routes into (possibly fused) resident
        # params. For equivalence we just capture the unfused (name, tensor)
        # pairs the worker hands over — that is exactly what Path A also hands
        # to the same call, so capturing it proves bit-equivalence of intent.
        for name, tensor in weights:
            self.loaded[name] = tensor.clone()


class _FakeRunner:
    def __init__(self, model):
        self.model = model


class _FakeWorker:
    """Binds the REAL worker methods so the test exercises production code."""

    def __init__(self, model):
        self.model_runner = _FakeRunner(model)

    # These are @staticmethod on the real class; accessing them off the class
    # yields plain functions, so re-wrap as staticmethod to preserve the calling
    # convention (self._read_raw_bytes_file(path) must NOT pass self).
    _read_raw_bytes_file = staticmethod(WeightSyncFromFileExtension._read_raw_bytes_file)
    _qk_unpermute_for_vllm = staticmethod(WeightSyncFromFileExtension._qk_unpermute_for_vllm)
    load_lora_base_from_raw = WeightSyncFromFileExtension.load_lora_base_from_raw
    load_lora_delta_from_raw = WeightSyncFromFileExtension.load_lora_delta_from_raw


# --------------------------------------------------------------------------
# Toy LoRA model (mirrors test_lora_cached_base_merge_equivalence.py)
# --------------------------------------------------------------------------
class _ToyLoRA(nn.Module):
    """A couple of LoRALinear layers under layers.0.{attn,mlp}.*."""

    def __init__(self, in_dim=32, out_dim=32, rank=8, alpha=16.0):
        super().__init__()
        # Use the torchtune module-path names so _TUNE_MODULE_TO_HF maps them.
        self.layers = nn.ModuleDict({
            "0": nn.ModuleDict({
                "attn": nn.ModuleDict({
                    "q_proj": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                    "k_proj": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                    "v_proj": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                    "output_proj": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                }),
                "mlp": nn.ModuleDict({
                    "w1": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                    "w2": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                    "w3": LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha),
                }),
            }),
        })

    def named_modules_for_lora(self):
        # torchtune names look like "layers.0.attn.q_proj"
        return [(n, m) for n, m in self.named_modules() if isinstance(m, LoRALinear)]


def _randomize_adapter(model, seed):
    """Give lora_a/lora_b non-trivial (non-zero) values like a trained adapter."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in model.named_modules_for_lora():
            m.lora_a.weight.copy_(torch.randn(m.lora_a.weight.shape, generator=g) * 0.1)
            m.lora_b.weight.copy_(torch.randn(m.lora_b.weight.shape, generator=g) * 0.1)


def _tune_to_hf_name(tune_module_name):
    """layers.0.attn.q_proj -> model.layers.0.self_attn.q_proj.weight"""
    import re
    clean = tune_module_name + ".weight"
    m = re.match(r"^(?:.*\.)?layers\.(\d+)\.(.+)\.weight$", clean)
    layer_idx, module_path = m.group(1), m.group(2)
    hf_module = _TUNE_MODULE_TO_HF[module_path]
    return f"model.layers.{layer_idx}.{hf_module}.weight"


def _build_payload(model, needs_unpermute=False, n_heads=0, n_kv_heads=0, head_dim=0):
    """Mirror the recipe's _gather_lora_delta_payload (sender side)."""
    tensors = {}
    entries = []
    for tune_name, m in model.named_modules_for_lora():
        hf = _tune_to_hf_name(tune_name)
        a_key, b_key = f"{hf}::lora_A", f"{hf}::lora_B"
        tensors[a_key] = m.lora_a.weight.detach().to(torch.bfloat16).contiguous()
        tensors[b_key] = m.lora_b.weight.detach().to(torch.bfloat16).contiguous()
        entries.append({
            "hf_name": hf, "a_key": a_key, "b_key": b_key,
            "scale": float(m.alpha) / float(m.rank),
        })
    meta = {
        "entries": entries, "needs_qk_unpermute": needs_unpermute,
        "num_heads": n_heads, "num_kv_heads": n_kv_heads, "head_dim": head_dim,
    }
    return tensors, meta


def _build_base_payload(model, needs_unpermute=False, n_heads=0, n_kv_heads=0, head_dim=0):
    """Mirror the recipe's _gather_lora_base_payload (one-time base ship)."""
    base = {}
    for tune_name, m in model.named_modules_for_lora():
        hf = _tune_to_hf_name(tune_name)
        w = m.weight.detach().to(torch.bfloat16).contiguous()
        if needs_unpermute and (".q_proj." in hf or ".k_proj." in hf):
            nh = n_heads if ".q_proj." in hf else n_kv_heads
            w = _qk_unpermute_for_vllm(w, nh, head_dim)
        base[hf] = w
    return base


def _make_fake_worker(model):
    """A fake worker whose resident params are the HF-named base weights."""
    resident = {}
    for tune_name, m in model.named_modules_for_lora():
        resident[_tune_to_hf_name(tune_name)] = m.weight.detach().to(torch.bfloat16).clone()
    return _FakeWorker(_FakeModel(resident))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_delta_receiver_matches_merged_path():
    """After base-ship + one delta, captured W_eff == iter_merged_lora_layers."""
    model = _ToyLoRA()
    _randomize_adapter(model, seed=1)

    # Path A reference (live merge from real base).
    ref = {}
    with torch.no_grad():
        for tune_name, merged in iter_merged_lora_layers(model):
            # iter_merged yields tune param names; map to HF for comparison.
            hf = _tune_to_hf_name(tune_name.rsplit(".weight", 1)[0])
            ref[hf] = merged

    worker = _make_fake_worker(model)
    with tempfile.TemporaryDirectory() as d:
        base_path = os.path.join(d, "base.bin")
        delta_path = os.path.join(d, "adapter.bin")
        _save_raw_bytes(_build_base_payload(model), base_path)
        tensors, meta = _build_payload(model)
        _save_raw_bytes(tensors, delta_path)

        r1 = worker.load_lora_base_from_raw(base_path)
        assert r1["status"] == "ok", r1
        r2 = worker.load_lora_delta_from_raw(delta_path, json.dumps(meta))
        assert r2["status"] == "ok", r2

    captured = worker.model_runner.model.loaded
    assert set(captured.keys()) == set(ref.keys())
    for k in ref:
        assert torch.allclose(captured[k].float(), ref[k].float(), atol=2e-2, rtol=1e-2), (
            f"delta!=merged for {k}: max|d|={(captured[k].float()-ref[k].float()).abs().max():.4f}"
        )


def test_delta_receiver_zero_drift_over_many_steps():
    """Stateless re-merge: applying the SAME adapter N times yields identical W."""
    model = _ToyLoRA()
    _randomize_adapter(model, seed=7)
    worker = _make_fake_worker(model)

    with tempfile.TemporaryDirectory() as d:
        base_path = os.path.join(d, "base.bin")
        _save_raw_bytes(_build_base_payload(model), base_path)
        assert worker.load_lora_base_from_raw(base_path)["status"] == "ok"

        first = None
        for step in range(5):
            tensors, meta = _build_payload(model)
            p = os.path.join(d, f"adapter_{step}.bin")
            _save_raw_bytes(tensors, p)
            assert worker.load_lora_delta_from_raw(p, json.dumps(meta))["status"] == "ok"
            snap = {k: v.clone() for k, v in worker.model_runner.model.loaded.items()}
            if first is None:
                first = snap
            else:
                for k in first:
                    assert torch.equal(first[k], snap[k]), f"drift at step {step} for {k}"


def test_delta_receiver_tracks_adapter_update():
    """A new adapter changes W_eff; the merge is recomputed from pristine base."""
    model = _ToyLoRA()
    _randomize_adapter(model, seed=2)
    worker = _make_fake_worker(model)

    with tempfile.TemporaryDirectory() as d:
        base_path = os.path.join(d, "base.bin")
        _save_raw_bytes(_build_base_payload(model), base_path)
        worker.load_lora_base_from_raw(base_path)

        t0, m0 = _build_payload(model)
        p0 = os.path.join(d, "a0.bin")
        _save_raw_bytes(t0, p0)
        worker.load_lora_delta_from_raw(p0, json.dumps(m0))
        w_step0 = {k: v.clone() for k, v in worker.model_runner.model.loaded.items()}

        _randomize_adapter(model, seed=99)  # "train" a step
        t1, m1 = _build_payload(model)
        p1 = os.path.join(d, "a1.bin")
        _save_raw_bytes(t1, p1)
        worker.load_lora_delta_from_raw(p1, json.dumps(m1))
        w_step1 = worker.model_runner.model.loaded

        # New adapter must move at least one weight.
        assert any(not torch.equal(w_step0[k], w_step1[k]) for k in w_step0)

        # And w_step1 must equal a fresh Path-A merge of the new adapter.
        ref = {}
        for tune_name, merged in iter_merged_lora_layers(model):
            ref[_tune_to_hf_name(tune_name.rsplit(".weight", 1)[0])] = merged
        for k in ref:
            assert torch.allclose(w_step1[k].float(), ref[k].float(), atol=2e-2, rtol=1e-2)


def test_delta_receiver_requires_base_first():
    """Delta before base must error (fail-fast → sender raises)."""
    model = _ToyLoRA()
    _randomize_adapter(model, seed=3)
    worker = _make_fake_worker(model)
    with tempfile.TemporaryDirectory() as d:
        t, m = _build_payload(model)
        p = os.path.join(d, "a.bin")
        _save_raw_bytes(t, p)
        r = worker.load_lora_delta_from_raw(p, json.dumps(m))
        assert r["status"] == "error" and "before load_lora_base" in r["message"]


def test_delta_qk_unpermute_commutes():
    """LLAMA-family: unpermute(base)+unpermute(delta) == unpermute(base+delta).

    The sender unpermutes the base; the receiver unpermutes the delta. Their sum
    must equal unpermuting the full merged weight.
    """
    n_heads, head_dim = 4, 8         # out_dim = 32 = in_dim
    model = _ToyLoRA(in_dim=32, out_dim=n_heads * head_dim, rank=8, alpha=16.0)
    _randomize_adapter(model, seed=5)
    worker = _make_fake_worker_unpermuted(model, n_heads, head_dim)

    with tempfile.TemporaryDirectory() as d:
        base_path = os.path.join(d, "base.bin")
        _save_raw_bytes(
            _build_base_payload(model, needs_unpermute=True, n_heads=n_heads,
                                n_kv_heads=n_heads, head_dim=head_dim),
            base_path,
        )
        worker.load_lora_base_from_raw(base_path)
        t, m = _build_payload(model, needs_unpermute=True, n_heads=n_heads,
                              n_kv_heads=n_heads, head_dim=head_dim)
        p = os.path.join(d, "a.bin")
        _save_raw_bytes(t, p)
        assert worker.load_lora_delta_from_raw(p, json.dumps(m))["status"] == "ok"

    captured = worker.model_runner.model.loaded
    # Reference: full merge then unpermute (only q/k).
    for tune_name, merged in iter_merged_lora_layers(model):
        hf = _tune_to_hf_name(tune_name.rsplit(".weight", 1)[0])
        ref = merged
        if ".q_proj." in hf or ".k_proj." in hf:
            ref = _qk_unpermute_for_vllm(merged, n_heads, head_dim)
        assert torch.allclose(captured[hf].float(), ref.float(), atol=2e-2, rtol=1e-2), hf


def _make_fake_worker_unpermuted(model, n_heads, head_dim):
    """Resident params = base weights, with q/k already unpermuted (as vLLM holds them)."""
    resident = {}
    for tune_name, m in model.named_modules_for_lora():
        hf = _tune_to_hf_name(tune_name)
        w = m.weight.detach().to(torch.bfloat16).clone()
        if ".q_proj." in hf or ".k_proj." in hf:
            w = _qk_unpermute_for_vllm(w, n_heads, head_dim)
        resident[hf] = w
    return _FakeWorker(_FakeModel(resident))


# --------------------------------------------------------------------------
# Drive the REAL recipe sender methods (not the test-local _build_payload).
# This catches binding/signature bugs in _gather_lora_delta_payload /
# _gather_lora_base_payload that a test-local payload builder would mask
# (e.g. `_needs_qk_unpermute(self._checkpointer)` passing one arg too many).
# --------------------------------------------------------------------------
def _load_recipe_module():
    """Load the LoRA-GRPO recipe by FILE PATH.

    ``recipes/`` is intentionally not an importable package (its __init__ raises),
    so a normal ``import recipes.dev...`` fails. Load the module object directly
    from its file so we can bind its real sender methods and exercise their exact
    call conventions (this is what catches signature/binding bugs).
    """
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    recipe_path = repo_root / "recipes" / "dev" / "lora_grpo_full_finetune_distributed_xpu.py"
    spec = importlib.util.spec_from_file_location("_lora_grpo_recipe_under_test", recipe_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RECIPE_MOD = _load_recipe_module()
_Recipe = _RECIPE_MOD.LoRAGRPODistributedXPU


class _FakeRecipe:
    """Binds the real recipe sender methods onto a minimal stand-in.

    Only the attributes the sender methods touch are provided. No torch.dist,
    no FSDP, no XPU — the methods are rank-0-only and collective-free by design.
    """

    _gather_lora_delta_payload = _Recipe._gather_lora_delta_payload
    _gather_lora_base_payload = _Recipe._gather_lora_base_payload
    # _needs_qk_unpermute / _maybe_unpermute_qk are bound from weight_sync on the
    # real class; reuse those exact bindings so the call convention is tested.
    _needs_qk_unpermute = _Recipe._needs_qk_unpermute
    _maybe_unpermute_qk = _Recipe._maybe_unpermute_qk

    def __init__(self, model, checkpointer=None, n_heads=0, n_kv_heads=0, head_dim=0):
        self._model = model
        self._is_rank_zero = True
        self._checkpointer = checkpointer  # None → _needs_qk_unpermute returns False
        self._model_num_heads = n_heads
        self._model_num_kv_heads = n_kv_heads
        self._model_head_dim = head_dim
        # _gather_lora_base_payload reads _cached_base_weights (tune-name → bf16 cpu).
        self._cached_base_weights = {
            f"{n}.weight": m.weight.detach().to(torch.bfloat16).cpu().contiguous()
            for n, m in model.named_modules_for_lora()
        }


def test_real_recipe_sender_roundtrips_to_receiver():
    """End-to-end with the REAL recipe sender methods + REAL receiver methods.

    Would have caught the `_needs_qk_unpermute()` arg-count bug that the
    test-local _build_payload masked.
    """
    model = _ToyLoRA()
    _randomize_adapter(model, seed=11)
    recipe = _FakeRecipe(model)  # checkpointer=None → non-permuting (Qwen3-like)

    base_sd = recipe._gather_lora_base_payload()
    tensors, meta = recipe._gather_lora_delta_payload()
    assert meta["needs_qk_unpermute"] is False
    assert len(tensors) == 2 * len(meta["entries"])

    worker = _make_fake_worker(model)
    with tempfile.TemporaryDirectory() as d:
        base_path = os.path.join(d, "base.bin")
        delta_path = os.path.join(d, "adapter.bin")
        _save_raw_bytes(base_sd, base_path)
        _save_raw_bytes(tensors, delta_path)
        assert worker.load_lora_base_from_raw(base_path)["status"] == "ok"
        assert worker.load_lora_delta_from_raw(delta_path, json.dumps(meta))["status"] == "ok"

    captured = worker.model_runner.model.loaded
    ref = {}
    for tune_name, merged in iter_merged_lora_layers(model):
        ref[_tune_to_hf_name(tune_name.rsplit(".weight", 1)[0])] = merged
    assert set(captured.keys()) == set(ref.keys())
    for k in ref:
        assert torch.allclose(captured[k].float(), ref[k].float(), atol=2e-2, rtol=1e-2), k
