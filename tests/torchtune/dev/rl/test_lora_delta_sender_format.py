"""Pin-down: the delta-path wire format round-trips and is ~66 MB, not 6.77 GiB.

Asserts that an adapter payload written by ``_save_raw_bytes`` (the sender's
transport) reads back through the worker's ``_read_raw_bytes_file`` with names,
shapes, dtype, and values preserved — and that the on-disk size is the small
adapter, not the full merged weight.
"""
from __future__ import annotations

import os
import tempfile

import torch

from torchtune.dev.rl.weight_sync import _save_raw_bytes
from torchtune.dev.vllm_weight_sync_worker import WeightSyncFromFileExtension


_read = WeightSyncFromFileExtension._read_raw_bytes_file


def test_adapter_payload_roundtrip():
    rank, in_dim, out_dim = 16, 2560, 2560
    tensors = {}
    for li in range(3):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hf = f"model.layers.{li}.self_attn.{proj}.weight"
            tensors[f"{hf}::lora_A"] = torch.randn(rank, in_dim, dtype=torch.bfloat16)
            tensors[f"{hf}::lora_B"] = torch.randn(out_dim, rank, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "adapter.bin")
        n = _save_raw_bytes(tensors, path)
        assert n == len(tensors)
        back = _read(path)

    assert set(back.keys()) == set(tensors.keys())
    for k, v in tensors.items():
        assert back[k].dtype == torch.bfloat16
        assert back[k].shape == v.shape
        assert torch.equal(back[k], v), f"value mismatch for {k}"


def test_adapter_payload_is_small():
    """A full Qwen3-4B adapter (252 tensors, r=16) is ~tens of MB, << 6.77 GiB."""
    rank = 16
    dims = {
        "q_proj": (2560, 4096), "k_proj": (1024, 4096), "v_proj": (1024, 4096),
        "o_proj": (4096, 2560), "gate_proj": (9728, 2560), "up_proj": (9728, 2560),
        "down_proj": (2560, 9728),
    }
    tensors = {}
    for li in range(36):
        for proj, (out_dim, in_dim) in dims.items():
            grp = "self_attn" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            hf = f"model.layers.{li}.{grp}.{proj}.weight"
            tensors[f"{hf}::lora_A"] = torch.zeros(rank, in_dim, dtype=torch.bfloat16)
            tensors[f"{hf}::lora_B"] = torch.zeros(out_dim, rank, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "adapter.bin")
        _save_raw_bytes(tensors, path)
        size_mb = os.path.getsize(path) / 1024**2

    assert len(tensors) == 36 * 7 * 2  # 504 adapter tensors
    # bf16 r=16 adapter: well under 200 MB; the merged W_eff would be ~6.77 GiB.
    assert size_mb < 200, f"adapter unexpectedly large: {size_mb:.1f} MB"
