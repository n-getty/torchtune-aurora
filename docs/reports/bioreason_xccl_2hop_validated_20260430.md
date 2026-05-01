# BioReason 2-node XCCL `node_fanout` (real 2-hop) — validated end-to-end

**Date**: 2026-04-30 / 2026-05-01
**Hold**: 8462930 (2 nodes, debug, 1h)
**Recipe**: `recipes/dev/grpo_bioreason_distributed_xpu.py`
**Config**: `recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml`
**Topology gates**: `WSYNC_METHOD=xccl`, `WSYNC_TOPOLOGY={replica_fanout|node_fanout}`

## Summary

Two unrelated fixes landed together on the BioReason 2-node Phase 2
launcher and were validated on the same hold:

1. **XCCL gather backbone-prefix fix** (root cause of the v5 hang originally
   misattributed to SSH-parent-death). `_sync_weights_to_vllm_xccl` in both
   FSDP1 and FSDP2 gather paths was sending `backbone.*` names to vLLM,
   which crashed `load_weights` with
   `ValueError: There is no module or parameter named 'backbone' in
   Qwen3ForCausalLM`. Train ranks then hung in `optimizer.step()` waiting
   on a bg gloo broadcast that the dead vLLM workers would never complete.

2. **`WSYNC_TOPOLOGY=node_fanout`** — the real 2-hop fanout (1 cross PG
   total, 1 intra PG of size `num_replicas * tp_size` on the vLLM node) is
   now validated as a drop-in replacement for the legacy `replica_fanout`
   (one cross PG per replica). Reduces cross-node traffic 12× for DP=12.

Both fixes are gated and default-safe: `replica_fanout` remains the default
topology, and the backbone-strip helper only triggers when
`self._is_bioreason` is True.

## Bug 1: backbone-prefix mismatch in XCCL gather paths

### Symptom

- v5 launch reached step 0 cleanly, then all 11 train ranks hung in
  `optimizer.step()` (futex_wait_queue). vLLM workers stayed alive but
  their `receive_weights_xccl_streaming` RPC handler had errored out and
  the failure never propagated back across the bg gloo broadcast.
- Originally attributed to the v8-style SSH-parent-death failure mode and
  patched with launcher-side `setsid + nohup + persistent watcher`. The
  hardening was correct (and shipped) but didn't fix the hang because the
  cause was elsewhere.

### Root cause

`torchtune/dev/rl/weight_sync.py:_sync_weights_to_vllm_xccl` builds
`hf_state_dict[hf_name]` from `self._tune_to_hf_map.get(param_name, param_name)`
on both the FSDP1 path (~line 1453) and the FSDP2 path (~line 1671). For
BioReason, the model is `BioReasonModel` whose Qwen3 backbone is wrapped
as `self.backbone`, so every param name carries a `backbone.` prefix
(e.g. `backbone.model.layers.0.mlp.gate_up_proj.weight`). vLLM only owns
the Qwen3 module and rejects `backbone.*` outright at
`vllm/model_executor/models/utils.py:326`.

The validated raw_bytes path already strips this via `_accept_and_rename`
at `weight_sync.py:995` — the XCCL path was simply missing the same
treatment.

### Fix

Added `_xccl_accept_and_rename` at the top of `_sync_weights_to_vllm_xccl`
mirroring the raw_bytes helper, and wrapped both gather sites:

```python
_is_bior = getattr(self, "_is_bioreason", False)

def _xccl_accept_and_rename(name: str):
    if _is_bior:
        clean = name.replace("_fsdp_wrapped_module.", "")
        clean = clean.replace("_checkpoint_wrapped_module.", "")
        if not clean.startswith("backbone."):
            return None
        return clean[len("backbone."):]
    return self._tune_to_hf_map.get(name, name)
```

FSDP1 path skips with `if hf_name is None: continue` before the device
copy. FSDP2 path skips before the `cast → metadata → batch` block, also
deleting the param tensor to free GPU memory promptly.

### Validation

Test A on hold 8462930: `WSYNC_METHOD=xccl WSYNC_TOPOLOGY=replica_fanout
NSTEPS=2 GRPO_SAMPLES=4`. 2/2 steps clean. XCCL communicator init in 41 ms
(world=13: 1 train + 12 vLLM), KL evolves 0.0019 → 0.0016, gen 30 s,
grpo 15-26 s, total 46.8 / 55.8 s.

## node_fanout: real 2-hop XCCL fanout

### Architecture

```
Train node:  rank 0 ── cross PG (size 2, gloo) ──→ vLLM rank 0
                                                  │
                                                  └── intra PG (size 12,
                                                      XCCL over XeLink)
                                                      vLLM ranks 0..11
```

`replica_fanout` keeps the legacy shape: one 2-rank cross PG per replica
(12 cross PGs for DP=12, broadcast in parallel). `node_fanout` collapses
the cross side to a single 2-rank PG and the intra side to a single PG of
`num_replicas * tp_size` on the vLLM node. Cross-node traffic drops from
`num_replicas × 61 GiB` to `1 × 61 GiB`; the intra-node hop fans out over
XeLink (~7-8 GB/s effective per rank).

Selected by `WSYNC_TOPOLOGY=node_fanout`. The launcher passes the env
through to both train ranks and to vLLM workers (via the
`init_xccl_communicator` POST). vLLM-side handler at
`torchtune/dev/vllm_weight_sync_worker.py:324` accepts `topology` +
`intra_world` args.

### Validation

Same hold 8462930, three runs:

| Test | Config | NSTEPS | Steps clean | Steady-state step time |
|------|--------|--------|-------------|------------------------|
| B    | node_fanout, G=4 fbs=4         | 2  | 2/2  | ~45 s |
| C    | node_fanout, G=4 fbs=4         | 10 | 10/10 | 37 s (gen 22.8 s, grpo 13-14 s after warmup) |
| D    | node_fanout + IPEX varlen, G=8 fbs=4 | 5  | 5/5  | ~49 s |

XCCL communicator init: ~30 ms in all three. node_fanout init log:
`node_fanout cross PG created (1 PG, method=gloo, intra_world=12 on
vLLM_NODE)`.

Test C step-by-step (steady state begins at step 4):

```
Step 1: gen=26.2 grpo=15.4 total=42.1 KL=0.0022
Step 2: gen=26.3 grpo=13.9 total=41.0 KL=0.0020
Step 3: gen=26.5 grpo=13.3 total=40.5 KL=0.0016
Step 4: gen=23.1 grpo=13.6 total=37.5 KL=0.0017
Step 5: gen=23.1 grpo=14.9 total=38.6 KL=0.0035
Step 6: gen=23.3 grpo=14.6 total=38.5 KL=0.0019
Step 7: gen=22.9 grpo=13.8 total=37.4 KL=0.0016
Step 8: gen=22.9 grpo=14.0 total=37.5 KL=0.0019
Step 9: gen=22.8 grpo=13.4 total=36.9 KL=0.0025
Step 10: gen=22.6 grpo=12.8 total=36.0 KL=0.0022
```

Test D (G=8, IPEX varlen) confirms the topology survives the doubled
rollout-side memory pressure that previously crashed BioReason in the
banned:1 PDE class:

```
Step 1: gen=33.5 grpo=17.1 total=50.7 KL=0.0030
Step 2: gen=32.7 grpo=15.9 total=49.0 KL=0.0029
Step 3: gen=32.2 grpo=14.5 total=47.2 KL=0.0021
Step 4: gen=31.6 grpo=14.6 total=46.7 KL=0.0024
Step 5: gen=34.0 grpo=16.8 total=51.5 KL=0.0036
```

G=8 over G=4: 1.32× wall for 2× rollout throughput, no OOM, no banned:1.

## Files touched

- `torchtune/dev/rl/weight_sync.py` — `_xccl_accept_and_rename` helper +
  applied at FSDP1 (`~line 1453`) and FSDP2 (`~line 1671`) gather sites.
- `torchtune/dev/rl/weight_sync.py:1308-1414` — node_fanout topology
  branch in `_setup_xccl_wsync_pg` (added prior session).
- `torchtune/dev/vllm_weight_sync_worker.py:320-502` — `topology` +
  `intra_world` args in `init_xccl_communicator` (added prior session).
- `experiments/bioreason/run_bioreason_2node_server.sh:453-457` — env
  passthrough for `WSYNC_TOPOLOGY` (added prior session).

## Production envelope after this validation

- BioReason 4B 2-node Phase 2 server mode now has a stable XCCL wsync
  path, end-to-end. Either `replica_fanout` (legacy, 12 cross PGs) or
  `node_fanout` (1 cross PG) works; `node_fanout` is the recommended path
  for DP≥4.
- G=8 with `TORCHTUNE_USE_IPEX_VARLEN=1` is the production envelope (2×
  rollout throughput vs G=4, 1.32× wall).
- Long-horizon stability validated to 10 steps at G=4 and 5 steps at G=8.
  No memory drift, KL stable, no banned:1.
