# vLLM XPU: PP KV-Cache Init Crashes with `KeyError` in `get_layers_from_vllm_config`

**Status**: ROOT CAUSE IDENTIFIED, WORKAROUND AVAILABLE (use TP=16/24 PP=1 instead of PP>1)

**Observed**: 2026-05-07, Aurora, vLLM 0.15.0 (frameworks/2025.3.1), Ray 2.53.0,
Qwen3-Coder-480B-A35B (Qwen3MoeForCausalLM), TP=8 PP=3, 2-node 24-tile deployment.

---

## Summary

When using `distributed_executor_backend="ray"` with pipeline parallelism (`PP > 1`) on
Intel XPU (Aurora), every vLLM worker crashes during KV cache initialization with a
`KeyError` on attention layer names that belong to a **different** pipeline stage:

```
KeyError: 'model.layers.0.self_attn.attn'   # PP stage 0 boundary
KeyError: 'model.layers.21.self_attn.attn'  # PP stage 1 boundary
KeyError: 'model.layers.42.self_attn.attn'  # PP stage 2 boundary
```

All 24 workers fail simultaneously. The model weights load successfully — this is a
post-load initialization failure, not a weight I/O problem.

**This is XPU-specific.** The CUDA path uses `vllm/v1/worker/gpu/model_runner.py` (tested
with PP+Ray). XPU uses a separate `vllm/v1/worker/gpu_model_runner.py` that has diverged
and does not carry the same PP correctness.

**Workaround**: For models where PP is only needed due to HBM capacity, check whether a
larger TP with PP=1 is feasible. Two constraints apply: `TP % num_kv_heads == 0` for KV
head replication, AND `vocab_size % TP == 0` for `VocabParallelEmbedding`. For
Qwen3-Coder-480B (num_kv_heads=8, vocab_size=151936=2^7×1187), only power-of-2 TP values
are valid. TP=16 across 2 nodes (PP=1) with `gpu_memory_utilization=0.98` fits at
~56 GB/tile and avoids this bug.

---

## Environment

- **Hardware**: Intel Data Center GPU Max 1550 (Ponte Vecchio), 64 GiB HBM/tile, 12 tiles/node
- **System**: ALCF Aurora HPC, 2-node PBS allocation
- **vLLM**: 0.15.0 (Aurora frameworks/2025.3.1)
- **Ray**: 2.53.0
- **PyTorch**: frameworks/2025.3.1
- **Model**: Qwen3-Coder-480B-A35B-Instruct (`Qwen3MoeForCausalLM`), BF16
- **Config**: TP=8, PP=3, 24 total workers, `enforce_eager=True`, `distributed_executor_backend="ray"`

---

## Full Traceback

```
(EngineCore_DP0 pid=209455) (RayWorkerWrapper pid=210870) ERROR
  File "vllm/v1/worker/worker_base.py", line 344, in execute_method
    self.model_runner.initialize_kv_cache(kv_cache_config)
  File "vllm/v1/worker/gpu_model_runner.py", line 5862, in initialize_kv_cache
    self.initialize_attn_backend(kv_cache_config)
  File "vllm/v1/worker/gpu_model_runner.py", line 5203, in initialize_attn_backend
    attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
  File "vllm/v1/worker/gpu_model_runner.py", line 5151, in get_attn_backends_for_group
    layers = get_layers_from_vllm_config(
  File "vllm/config/vllm.py", line 1504, in get_layers_from_vllm_config
    if isinstance(forward_context[layer_name], layer_type)
KeyError: 'model.layers.21.self_attn.attn'

RuntimeError: Engine core initialization failed.
```

All 24 workers fail with the same KeyError (on layer 0, 21, or 42 — the first layer of
each PP stage).

---

## Root Cause Analysis

### Background: how PP layer assignment works

`vllm/model_executor/models/utils.py::make_layers()` creates real `Qwen3MoeDecoderLayer`
objects only for the local PP stage's range (`start_layer..end_layer`) and
`PPMissingLayer()` placeholders for all other layers. Each real layer's `Attention.__init__`
registers itself in `vllm_config.compilation_config.static_forward_context[prefix]`
(e.g., `"model.layers.0.self_attn.attn"`). So after model init, a PP stage 0 worker's
`static_forward_context` contains only `model.layers.{0..20}.self_attn.attn`.

### The KV cache config path

`get_kv_cache_specs()` (called on each worker by the EngineCore) reads `static_forward_context`
and returns only local PP stage layers. `get_kv_cache_configs()` in `kv_cache_utils.py`
merges all workers' specs, builds global KV cache groups, then filters per-worker:

```python
group_layer_names_one_worker = [
    name for name in group.layer_names
    if name in kv_cache_spec_one_worker   # ← should give local-stage layers only
]
```

This filtering is correct in theory. But after `get_kv_cache_configs` assembles
`kv_cache_configs`, it dispatches them to workers via:

```python
self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
```

Each worker then uses `kv_cache_configs[self.global_rank]`.

### The divergence

The XPU path (`gpu_model_runner.py`) has diverged from the CUDA path
(`gpu/model_runner.py` + `gpu/attn_utils.py`) at some point in vLLM's development.
Exactly where the rank-to-config mapping goes wrong in the XPU path was not fully
traced, but the observable consequence is that one or more PP workers receive a
`kv_cache_config` whose `kv_cache_group_spec.layer_names` contains layer names from
a **different** PP stage than the worker's own `static_forward_context`.

### The immediate crash site

`get_layers_from_vllm_config` (`vllm/config/vllm.py:1504`) does an unsafe dict lookup:

```python
# Current (unsafe):
return {
    layer_name: forward_context[layer_name]
    for layer_name in layer_names
    if isinstance(forward_context[layer_name], layer_type)  # KeyError if absent
}

# Safe version:
return {
    layer_name: forward_context[layer_name]
    for layer_name in layer_names
    if layer_name in forward_context
    and isinstance(forward_context[layer_name], layer_type)
}
```

Making this safe would suppress the KeyError but not fix the upstream misconfiguration —
`get_attn_backends_for_group` then does `layers[layer_name].get_attn_backend()` for each
name in `kv_cache_group_spec.layer_names`, which would KeyError on the same absent layers.
The correct fix is either to repair the rank-to-config mapping, or to filter
`kv_cache_group_spec.layer_names` against `static_forward_context` before the inner call.

---

## Timeline / What Was Validated Before the Bug Hit

| Step | Result |
|------|--------|
| Ray 2-node cluster startup (TP=8 PP=3, world_size=24) | ✓ PASS |
| Ray worker spawning on both nodes | ✓ PASS |
| Lustre page-cache prewarm (895 GiB, 16 parallel readers/node) | ✓ PASS — 4.5–9 GiB/s |
| vLLM weight loading from warm cache | ✓ PASS — 647s (worker), 1001s (head) |
| KV cache initialization | ✗ FAIL — KeyError on cross-stage layer names |

The model does fully load into GPU memory before the crash. The bug is specifically in
the post-load `initialize_kv_cache` → `initialize_attn_backend` phase.

---

## Workarounds

### Primary (recommended): increase TP to avoid PP

For models where PP is driven by HBM capacity rather than architecture constraints,
check whether a larger TP with PP=1 fits:

- `TP ≤ num_kv_heads`: partition path, must divide evenly
- `TP > num_kv_heads`: **replication path** — `assert tp_size % num_kv_heads == 0`

For Qwen3-Coder-480B (num_kv_heads=8, 895 GB BF16):

| Config | Tiles | GB/tile | Fits 64 GB? | Valid? | Notes |
|--------|-------|---------|-------------|--------|-------|
| TP=8  PP=3 | 24 | 112 | ✗ | — | Weights don't fit; PP broken anyway |
| TP=16 PP=1 | 16 | 56  | ✓ (tight) | ✓ | util=0.98, max_model_len≤1024 |
| TP=24 PP=1 | 24 | 37  | ✓ | ✗ | 151936 % 24 ≠ 0 — vocab constraint |

`vocab_size=151936 = 2^7 × 1187` — factor of 3 in TP=24 is incompatible with
`VocabParallelEmbedding` which requires `vocab_size % tp_size == 0`. Only power-of-2
TP values are valid for this model. TP=16 PP=1 is the correct workaround.

### Secondary: monkey-patch `initialize_attn_backend`

If PP is unavoidable, patch the per-worker `kv_cache_group_spec.layer_names` to filter
against the local `static_forward_context` before `get_attn_backends_for_group` is called.
This must reach the **Ray worker processes** (not the driver). A user-level
`sitecustomize.py` with an import hook is one approach. Editing the installed
`gpu_model_runner.py` at the relevant cluster path (if writable) is more direct.

The fix in `gpu_model_runner.py::initialize_attn_backend` (around line 5202):

```python
local_ctx = self.vllm_config.compilation_config.static_forward_context
for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
    # PP fix: filter to layers present on this PP stage
    kv_cache_group_spec.layer_names = [
        n for n in kv_cache_group_spec.layer_names if n in local_ctx
    ]
    attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
    ...
```

`KVCacheGroupSpec.layer_names` is a plain `list[str]` field (dataclass), so it can be
replaced in-place.

---

## Upstream Filing

File against vLLM at https://github.com/vllm-project/vllm/issues with:
- Component: XPU / Pipeline Parallel
- Version: 0.15.0
- Reproduction: Ray executor, PP > 1, any model, XPU backend
- Attach this file and the full traceback above
- Note the CUDA path (`gpu/model_runner.py` + `gpu/attn_utils.py`) works; divergence
  is in `gpu_model_runner.py` (XPU-specific file)

ALCF-side: report to `support@alcf.anl.gov` for inclusion in the next frameworks release.
