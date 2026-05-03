"""CPU pin-down for EP wsync fp8 on-wire (WS8).

Pin-down for the cast/AG/decompress math behind
``TORCHTUNE_EP_WSYNC_FP8_WIRE=1`` in
``torchtune/dev/rl/weight_sync.py::_sync_weights_to_vllm_xccl``.

The default EP expert path AllGathers full bf16 expert shards over
``_shard_pg`` (~57 GiB / step at EP=16 → ~70s wsync_gather floor).
The opt-in path casts the local shard to fp8 (E4M3) with rowwise scales
along the last dim, AllGathers fp8 + scale, decompresses on the active
rank back to bf16, and feeds the existing per-layer streaming fuse
identically. Wire bytes drop ~2× (fp8 + 4-byte scale per output row).

Numerical contract (per feedback section #10): expert weights are the
only fp8 candidates; dense / shared / router / norm / attention tensors
are NOT touched by this path. Round-trip error is bounded but NOT
bit-exact — this test pins down the bound, not equality.

Pure-Python (no torch.distributed, no XPU); runs on a login node in ~1s.
"""
from __future__ import annotations

import pytest
import torch


# --- The functions under test (mirror weight_sync.py implementation) ----

def _fp8_quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast bf16 expert shard to E4M3 with per-output-row scale.

    For shard shape ``[n_local_experts, out_dim, in_dim]``, the scale shape
    is ``[n_local_experts, out_dim, 1]`` (one scale per output row of each
    expert). Per-row scaling is the standard fp8 weight-quant convention
    and tracks long-tail magnitudes much better than per-tensor.
    """
    if x.dtype != torch.bfloat16 and x.dtype != torch.float16 and x.dtype != torch.float32:
        raise ValueError(f"unsupported dtype for fp8 quant: {x.dtype}")
    x_f = x.to(torch.float32)
    if x_f.ndim < 2:
        raise ValueError(f"expert shard must be >= 2D, got {x_f.shape}")
    amax = x_f.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 448.0).clamp(min=1e-12)
    x_scaled = (x_f / scale).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale.to(torch.float32)


def _fp8_dequantize_rowwise(x_fp8: torch.Tensor, scale: torch.Tensor,
                            out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Inverse of ``_fp8_quantize_rowwise``."""
    return (x_fp8.to(torch.float32) * scale).to(out_dtype)


# --- Tests ---------------------------------------------------------------

@pytest.mark.parametrize(
    "n_local, intermediate, hidden",
    [
        (4, 768, 2048),       # Qwen3-30B-A3B EP=16 gate/up shape (per rank)
        (8, 768, 2048),       # Qwen3-30B-A3B EP=8
        (4, 2048, 768),       # down_proj shape
        (16, 64, 64),         # tiny smoke
    ],
)
def test_fp8_roundtrip_error_bounded(n_local, intermediate, hidden):
    """Round-trip max abs err must be bounded by the row's own amax.

    E4M3 has 3 mantissa bits → max relative quant error per element is
    ~1/8 (1 LSB at the largest exponent for the row's amax). After
    multiplying by the per-row scale, the per-element absolute error is
    bounded by ``row_amax / 8``. We assert a slightly looser bound
    (``row_amax / 4``) to allow for the secondary bf16 cast at
    decompression — this catches scale-axis or clamp regressions that
    blow err up to ``row_amax`` order or higher.
    """
    torch.manual_seed(42)
    x = torch.randn(n_local, intermediate, hidden, dtype=torch.bfloat16) * 0.02
    x_fp8, scale = _fp8_quantize_rowwise(x)
    x_back = _fp8_dequantize_rowwise(x_fp8, scale, torch.bfloat16)

    assert x_fp8.shape == x.shape, f"fp8 shape mismatch: {x_fp8.shape} vs {x.shape}"
    assert scale.shape == (n_local, intermediate, 1), scale.shape
    assert x_back.shape == x.shape

    err = (x.float() - x_back.float()).abs()
    row_amax = x.float().abs().amax(dim=-1, keepdim=True)
    # Per-element bound: row_amax / 4 (≥ 2× the theoretical E4M3 LSB).
    bound = (row_amax / 4.0).expand_as(err)
    n_violations = (err > bound).sum().item()
    assert n_violations == 0, (
        f"{n_violations} elements exceed row_amax/4 bound; "
        f"max err {err.max().item():.6f}, max row_amax {row_amax.max().item():.6f}"
    )


@pytest.mark.parametrize("n_local, intermediate, hidden", [(8, 768, 2048)])
def test_fp8_roundtrip_mean_rel_err_under_3pct(n_local, intermediate, hidden):
    """Mean relative error across non-tiny entries should sit at typical
    fp8-weight-quant levels (~2% on Gaussian). Catches catastrophic
    regressions like clamp-only or wrong scale axis.
    """
    torch.manual_seed(7)
    x = torch.randn(n_local, intermediate, hidden, dtype=torch.bfloat16) * 0.02
    x_fp8, scale = _fp8_quantize_rowwise(x)
    x_back = _fp8_dequantize_rowwise(x_fp8, scale, torch.bfloat16)

    err = (x.float() - x_back.float()).abs()
    mask = x.float().abs() > 1e-3
    rel = err[mask] / x.float().abs()[mask]
    assert rel.mean().item() < 0.03, f"mean rel err {rel.mean().item():.4f} > 3%"


def test_fp8_quant_dequant_zero_input_is_zero():
    """All-zero input round-trips to all-zero (no NaN from 0/0 scale)."""
    x = torch.zeros(4, 16, 32, dtype=torch.bfloat16)
    x_fp8, scale = _fp8_quantize_rowwise(x)
    x_back = _fp8_dequantize_rowwise(x_fp8, scale, torch.bfloat16)
    assert torch.equal(x_back, x), "all-zero input must round-trip exactly"
    assert not torch.isnan(scale).any()
    assert not torch.isinf(scale).any()


def test_fp8_dtype_size_2x_compression():
    """Sanity: fp8 element_size == 1 byte; 2x vs bf16."""
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    x_fp8, _ = _fp8_quantize_rowwise(x)
    assert x_fp8.element_size() == 1
    assert x.element_size() == 2


# --- End-to-end: simulate AllGather + per-layer fuse path ---------------

def _build_local_shards(ep_d: int, n_local: int, intermediate: int,
                        hidden: int, n_layers: int, seed: int = 42):
    """Mirror ``test_ep_wsync_layer_batch_equivalence._build_local_shards``."""
    torch.manual_seed(seed)
    shards: list = []
    for r in range(ep_d):
        rank_dict: dict = {}
        for L in range(n_layers):
            rank_dict[L] = {
                "gate_proj": torch.randn(n_local, intermediate, hidden,
                                         dtype=torch.bfloat16) * 0.02,
                "up_proj": torch.randn(n_local, intermediate, hidden,
                                       dtype=torch.bfloat16) * 0.02,
                "down_proj": torch.randn(n_local, hidden, intermediate,
                                         dtype=torch.bfloat16) * 0.02,
            }
        shards.append(rank_dict)
    return shards


def _bf16_path(local_shards):
    """Reference: bf16 AllGather → unshuffle → full per-projection tensor.

    Mirrors the per-projection AG path at weight_sync.py:1952-1962.
    """
    ep_d = len(local_shards)
    out: dict = {}
    for L in local_shards[0]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            parts = [local_shards[r][L][proj] for r in range(ep_d)]
            stk = torch.stack(parts, dim=0)
            full = stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()
            out.setdefault(L, {})[proj] = full
    return out


def _fp8_path(local_shards):
    """fp8 cast → AllGather (fp8 + scale) → dequant on active rank →
    unshuffle → full per-projection tensor (bf16).
    """
    ep_d = len(local_shards)
    out: dict = {}
    for L in local_shards[0]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            fp8_parts = []
            scale_parts = []
            for r in range(ep_d):
                fp8_r, sc_r = _fp8_quantize_rowwise(local_shards[r][L][proj])
                fp8_parts.append(fp8_r)
                scale_parts.append(sc_r)
            # active rank decompresses each rank's part separately, then
            # stacks (matches the in-recipe ordering).
            bf16_parts = [
                _fp8_dequantize_rowwise(f, s, torch.bfloat16)
                for f, s in zip(fp8_parts, scale_parts)
            ]
            stk = torch.stack(bf16_parts, dim=0)
            full = stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()
            out.setdefault(L, {})[proj] = full
    return out


def test_fp8_path_unshuffle_matches_bf16_shape_and_layout():
    """fp8 round-trip preserves the interleaved expert ordering exactly
    (the bytes change, the structure does not).
    """
    ep_d, n_local = 4, 4
    shards = _build_local_shards(ep_d, n_local, 8, 16, n_layers=2)
    bf = _bf16_path(shards)
    fp = _fp8_path(shards)
    for L in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert bf[L][proj].shape == fp[L][proj].shape, (
                f"shape mismatch L={L} {proj}")
            # Per-expert slices end up in the same global slot.
            for global_i in range(ep_d * n_local):
                rank = global_i % ep_d
                local_i = global_i // ep_d
                # The fp8 path's expert at slot `global_i` should be a
                # round-trip of the original; bf16 path is the original.
                bf_slice = bf[L][proj][global_i]
                fp_slice = fp[L][proj][global_i]
                # bound: row_amax/128 (looser than per-row test because
                # we fold in a second cast at decompression).
                row_amax = bf_slice.float().abs().amax(dim=-1, keepdim=True)
                err = (bf_slice.float() - fp_slice.float()).abs()
                # Bound per element: row_amax / 4 (E4M3 LSB headroom + bf16 recast).
                bound = (row_amax / 4.0).expand_as(err)
                violations = (err > bound).sum().item()
                assert violations == 0, (
                    f"L={L} {proj} expert={global_i}: {violations} elements "
                    f"violate row_amax/4 bound; "
                    f"max err {err.max().item():.6f}, "
                    f"max row_amax {row_amax.max().item():.6f}"
                )


def test_fp8_wire_bytes_under_55pct_of_bf16():
    """Wire-bytes accounting (per-projection): fp8 + scale must be < 55%
    of the bf16 baseline. Captures the actual saving achievable on
    `_shard_pg`.
    """
    n_local, intermediate, hidden = 8, 768, 2048
    x = torch.randn(n_local, intermediate, hidden, dtype=torch.bfloat16) * 0.02
    x_fp8, scale = _fp8_quantize_rowwise(x)
    bf16_bytes = x.numel() * 2
    fp8_bytes = x_fp8.numel() * 1 + scale.numel() * 4
    ratio = fp8_bytes / bf16_bytes
    assert ratio < 0.55, (
        f"fp8 wire-bytes ratio {ratio:.4f} > 0.55; "
        f"scale overhead too high for this shape"
    )
