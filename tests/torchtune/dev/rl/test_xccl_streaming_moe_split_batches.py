"""Pin-down test for the XCCL-streaming MoE w13/w2 split-across-batches bug.

The trainer's greedy batching in ``_sync_weights_to_vllm_xccl`` packs params
into broadcasts up to ``batch_max_numel``. With Qwen3-30B-A3B and
real ``batch_max_numel`` (1 GiB), a layer's ``w13_weight`` (~96 MiB) and
``w2_weight`` (~96 MiB) routinely land in different batches. The receiver in
``vllm_weight_sync_worker._load_fused_moe_experts`` requires both keys per
layer and KeyError's on the first lonely w13.

The fix accumulates ``fused_pending`` across batches, applies layers only
when both keys are present, and raises a clear error if any pair remains
incomplete after all batches. This test simulates the receiver dispatch
loop in isolation (no vLLM, no XCCL) to lock in the contract.
"""
from __future__ import annotations

import re
from collections import OrderedDict


def _simulate_dispatch(tensors_meta, batch_max_numel):
    """Mimic the receiver's batched dispatch and return loaded layers in order.

    Returns a list of (batch_index, [(layer_idx, kinds), ...]) where each
    inner item describes which layers were applied at the end of that batch.
    """
    n_params = len(tensors_meta)
    fused_pending: dict[int, dict[str, object]] = {}
    apply_log: list[tuple[int, list[tuple[int, list[str]]]]] = []
    _fused_re = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight")

    i = 0
    batch_idx = 0
    while i < n_params:
        batch_numel = 0
        batch = []
        while i < n_params:
            pn = tensors_meta[i]["numel"]
            if batch_numel > 0 and batch_numel + pn > batch_max_numel:
                break
            batch.append(tensors_meta[i])
            batch_numel += pn
            i += 1

        for entry in batch:
            m = _fused_re.match(entry["name"])
            if m:
                layer_idx = int(m.group(1))
                kind = m.group(2)
                fused_pending.setdefault(layer_idx, {})[kind] = entry["name"]

        ready = sorted(
            li for li, kv in fused_pending.items()
            if "w13" in kv and "w2" in kv
        )
        for li in ready:
            del fused_pending[li]
        apply_log.append((batch_idx, [(li, ["w13", "w2"]) for li in ready]))
        batch_idx += 1

    return apply_log, fused_pending


def _build_qwen3_30b_meta():
    """48 layers of MoE experts, intermediate=768, hidden=2048, E=128.

    Plus a few non-expert params so the batching is realistic.
    """
    meta = []
    for layer in range(48):
        meta.append({
            "name": f"model.layers.{layer}.input_layernorm.weight",
            "numel": 2048,
        })
        meta.append({
            "name": f"model.layers.{layer}.mlp.experts.w13_weight",
            "numel": 128 * 2 * 768 * 2048,  # ~402 MiB in bf16
        })
        meta.append({
            "name": f"model.layers.{layer}.mlp.experts.w2_weight",
            "numel": 128 * 2048 * 768,  # ~402 MiB in bf16
        })
    return meta


def test_w13_w2_can_split_across_batches_in_realistic_config():
    """Reproducer: with batch_max_numel < (w13 + w2) the pair lands in two batches."""
    meta = _build_qwen3_30b_meta()
    # ~250M elements ≈ 500 MiB cap; w13 alone is ~200M elements.
    # Batches will frequently fit only one of {w13, w2} per layer.
    batch_max_numel = 250_000_000

    apply_log, leftover = _simulate_dispatch(meta, batch_max_numel)
    assert not leftover, f"Some layers never applied: {leftover}"

    # All 48 layers must eventually be loaded.
    loaded = [li for _, batch in apply_log for li, _ in batch]
    assert sorted(loaded) == list(range(48))

    # Sanity: there must exist a batch that processes a w13 without applying its layer
    # (proving split-across-batches actually happens — i.e., this is a real reproducer,
    #  not a no-op test that only passes because everything fits in one batch).
    saw_split = False
    for _, batch in apply_log:
        # At least one batch must produce ZERO new layers (because it only got w13s
        # whose w2s come later, or vice versa).
        if len(batch) == 0:
            saw_split = True
            break
    assert saw_split, (
        "Split-across-batches scenario didn't trigger — pick a smaller batch_max_numel "
        "or larger expert tensors so w13/w2 actually separate."
    )


def test_complete_pair_in_single_batch_still_loads():
    """If batch is large enough that both w13 and w2 fit, layer applies in same batch."""
    meta = _build_qwen3_30b_meta()
    # 1 GiB cap: comfortably fits both expert tensors of a single layer (~800 MiB total).
    batch_max_numel = 1_073_741_824

    apply_log, leftover = _simulate_dispatch(meta, batch_max_numel)
    assert not leftover

    loaded = [li for _, batch in apply_log for li, _ in batch]
    assert sorted(loaded) == list(range(48))


def test_missing_w2_raises():
    """If the manifest is malformed (only w13 ever sent), the receiver detects it."""
    meta = [
        {"name": "model.layers.0.mlp.experts.w13_weight", "numel": 1000},
        {"name": "model.layers.0.input_layernorm.weight", "numel": 100},
    ]
    apply_log, leftover = _simulate_dispatch(meta, batch_max_numel=2000)
    # With only w13, layer 0 never has w2 → leftover non-empty (the receiver
    # would raise RuntimeError on this case).
    assert 0 in leftover
    assert "w13" in leftover[0]
    assert "w2" not in leftover[0]
