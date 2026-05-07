# Qwen3-30B-A3B MoE — Next Steps

Created: 2026-04-27

## Current Baseline

- 10+2 tiles, single node, SHM weight sync + GPU transpose + GPU fuse
- **35.3s/step** steady-state (batch=1, grpo_samples=4, max_gen=64)
- GRPO fwd+bwd: ~23s (comparable to 32B dense on 10 tiles)
- Memory: FLAT at 30.41 GiB

## Core Problem

MoE has **11.5x less compute** per token than 32B dense, but **identical FSDP
communication volume** (~122 GB AllGather+ReduceScatter per fwd+bwd for 30B params).
On 10 tiles with XCCL, communication dominates (~15s of 23s). MoE's compute savings
are masked by communication cost.

Additionally, `torch.bmm` on [128, ~28, 2048] × [128, 2048, 768] is poorly saturated —
each per-expert matrix is small at current batch sizes.

---

## Tier 1: Immediate (config tuning)

### 1.1 Increase batch_size and/or grpo_samples

**Goal**: More tokens per step → better amortization of FSDP communication.

| Config | tokens/step (est.) | tokens/expert | Notes |
|--------|-------------------|---------------|-------|
| batch=1, grpo_samples=4, max_gen=64 | ~456 | ~28 | Current test config |
| batch=1, grpo_samples=4, max_gen=512 | ~2248 | ~140 | Current production config |
| batch=2, grpo_samples=4, max_gen=512 | ~4496 | ~281 | Test on current node |
| batch=1, grpo_samples=8, max_gen=512 | ~4496 | ~281 | More RL samples, same compute |
| batch=4, grpo_samples=4, max_gen=512 | ~8992 | ~562 | May need multi-node for memory |

Memory budget: 30B model on 10 tiles = 3B params/tile (~6 GiB). With AC enabled and
2048 embed_dim (vs 5120 for 32B dense), activations per token are ~6x smaller.
batch=2 should fit on single node; batch=4 worth testing.

**Expected outcome**: Step time stays ~23-25s (comm-bound) but throughput per step
increases proportionally with batch size. Effective tokens/second improves.

**Action**: Test batch=2 on current single-node setup, then grpo_samples=8.

### 1.2 Increase forward_batch_size (fbs)

Current fbs=2 means the 4 GRPO samples are processed in 2 forward micro-batches.
With batch=1 × grpo_samples=4, fbs=4 would process all samples in one forward pass,
reducing Python loop overhead and improving GPU utilization.

**Action**: Set fbs=4 (or equal to grpo_samples) when testing batch size changes.

---

## Tier 2: Kernel Optimizations

### 2.1 torch.compile on expert forward

**Goal**: Fuse scatter → bmm → silu → multiply → bmm → gather into fewer kernels.

```python
# In model builder or recipe setup:
for layer in model.layers:
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
        layer.mlp.experts = torch.compile(layer.mlp.experts, backend="inductor")
```

**Constraints**:
- XPU compile is viable single-node only (multi-node deadlocks with oneCCL)
- The expert forward has data-dependent shapes (`max_count` varies per step) —
  may cause recompilation. Guard with `torch._dynamo.config.cache_size_limit`.
- Test with `TORCH_COMPILE_DISABLE=0` (currently disabled in some launchers)

**Risk**: Medium — XPU inductor backend may not support all ops in expert forward.
Fallback: `torch.compile(mode="reduce-overhead")` or compile only the BMM portion.

**Expected outcome**: 2-5x speedup on the expert forward compute portion (~8s → 2-4s).
Net step impact depends on how much of the 23s is expert compute vs FSDP communication.

### 2.2 Vectorize scatter/gather Python loops

Replace the 128-iteration Python for-loops in `GroupedExpertsHF.forward()` with
vectorized torch ops:

```python
# Current (Python loop, 128 iterations):
for e in range(E):
    c = int(count_list[e])
    if c > 0:
        x_padded[e, :c] = x[offset : offset + c]

# Vectorized alternative:
# Pre-compute cumulative offsets and use advanced indexing
cumsum = counts.cumsum(0)
offsets = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), cumsum[:-1]])
# Use torch.scatter_ with pre-built index tensor
```

**Effort**: Low (pure Python → torch ops).
**Expected outcome**: Minor — the loops are fast for 128 iterations with small copies.
Worth doing as cleanup but not a performance priority.

### 2.3 IPEX FusedMoE for training forward

IPEX's `_IPEXGatedMLPMOEXPU` is used in vLLM inference but not in training.
It fuses gate+silu+up multiplication and expert dispatch into a single kernel.

**Challenge**: IPEX FusedMoE has no backward pass implementation. Options:
1. Use IPEX kernel for forward, PyTorch autograd for backward (via `torch.autograd.Function`)
2. Use IPEX only in the `no_grad` forward passes (policy logprob, ref logprob) — saves
   2/3 of forward compute without needing backward support
3. Request backward support from Intel IPEX team

**Effort**: Medium-high.
**Expected outcome**: Significant for inference-heavy steps (policy+ref forward = 2/3 of
forward passes are `no_grad`). Could cut those from ~7s each to ~2-3s.

### 2.4 Intel Triton fused MoE kernel

`intel-xpu-backend-for-triton` can compile Triton kernels for XPU. A custom fused MoE
kernel (similar to vLLM's Triton FusedMoE) would:
- Eliminate padding waste (process only active token-expert pairs)
- Fuse gate+silu+up into a single kernel
- Support backward via Triton's autograd integration

**Effort**: High — requires custom kernel development and XPU Triton debugging.
**Expected outcome**: Best possible expert compute performance, but high development cost.
Defer until torch.compile and IPEX options are evaluated.

---

## Tier 3: Multi-Node Scaling

### 3.1 Two-node dedicated vLLM

```
Node 0: 12 tiles → vLLM (TP=2 or TP=4)
Node 1: 12 tiles → Training (FSDP=12)
```

**Benefits**:
- 12 training tiles instead of 10 (20% more sharding, less memory per tile)
- Room for larger batch sizes
- No colocated vLLM memory contention

**Weight sync options**:

| Method | Sync time | Long-run stability | Notes |
|--------|-----------|-------------------|-------|
| 2-hop XCCL cross-node | ~9s | **Crashes ~step 28** (CXI MR cache) | Same 32B failure mode |
| 2-hop gloo cross-PG + XCCL intra | ~47s | 20/20 clean (32B validated) | Slow but stable |
| SHM (node-local) | N/A | — | Requires vLLM on same node as rank 0 |

**Key constraint**: Cross-node XCCL weight sync leaks ~9 MiB/step CXI MR cache entries
(validated definitively in Run 19c, 2026-04-27: PG reset does NOT help). For runs >28
steps, gloo cross-PG is the only viable cross-node method.

**Gloo overhead mitigation**: The 47s gloo broadcast is for 32B (61 GiB). MoE is 57 GiB
(531 fused params) — similar size, so expect ~44s gloo broadcast. With max_gen=512,
generation takes 30-60s, so the gloo broadcast could partially overlap with generation
if the async flow is wired correctly.

**Alternative**: Keep vLLM colocated on training node (10+2 layout) and use node 1 as
a pure HSDP replica. This avoids cross-node weight sync entirely — SHM stays local.
Training uses HSDP (replicate=2, shard=11 per node). The cost: only 10 training tiles
per node (2 reserved for vLLM on node 0), less memory efficiency.

**Action**: Start with the HSDP colocated approach (avoids the cross-node sync problem).
Test: 2-node HSDP with 10+2 on node 0 + 12 training on node 1. If memory allows batch=2+,
the throughput gain from more RL samples per step may outweigh the gloo overhead.

### 3.2 MoE-specific multi-node weight sync optimization

For MoE models, weight sync only needs to transfer the **active expert parameters that
changed** — with a learning rate of 5e-6 and top-8 routing, most expert weight updates
are small. Delta compression or sparse sync could reduce the 57 GiB transfer:
- Only sync params with `grad.norm() > threshold`
- Or sync every N steps (current `weight_sync_interval` support)

`weight_sync_interval=2` already validated for 32B (13.4% throughput improvement, halves
CXI leak rate). For MoE with fast generation, interval=2-4 is algorithmically sound and
extends safe XCCL run length to ~56-112 steps.

---

## Tier 4: Expert Parallelism (must-do, currently blocked)

### Current status

EP=4/DP=3 for Gemma4 26B-A4B reached v161 (forward works, backward blocked).
The same EP infrastructure applies to Qwen3-30B-A3B.

**Why EP matters for MoE performance**:
- Eliminates FSDP AllGather for expert params (each rank holds its own 1/EP expert shard)
- Reduces communication from ~122 GB (full model AG+RS) to ~13 GB (attention+router only)
- This is the only path to realizing MoE's 11.5x compute advantage

### The autograd ordering blocker

Two distinct bugs identified across v141-v161:

1. **Reentrant AC scheduling desync (v153)**: Ranks within an EP group execute backward
   ops in different orders under `use_reentrant=True`. Local-index-1 rank consistently
   one op behind peers. Gloo matches collectives by call order → mismatch → crash.

2. **Router non-determinism under non-reentrant AC (v154)**: Switching to
   `use_reentrant=False` fixes the scheduling desync but exposes a new bug:
   AC recompute regenerates router outputs (sigmoid → argsort → topk), which can
   differ by ±1 token at tie boundaries. The recomputed `num_tokens_per_expert`
   mismatches the original → `ScatterAddBackward0` shape error.

### Fix approaches (ordered by cost/risk)

1. **Cache router outputs through AC** (~30 lines in `_parallelism.py`)
   Save `_ag_gather_idx` and `_ag_s_local` during original forward, restore during
   AC recompute. Fixes the shape mismatch without changing router determinism.
   Risk: per-instance tensor side-channel through backward pass.

2. **Mark router outputs as saved tensors** (custom `checkpoint_fn`)
   Force AC to keep `(scores, selected_experts, num_tokens_per_expert)` from original
   forward instead of recomputing. Fixes determinism at source. Memory cost: O(T × E)
   per layer × 48 layers.

3. **Move EP dispatch outside autograd** (largest diff, cleanest fix)
   Restructure `_token_dispatch`/`_token_combine` as side-effecting operations with
   manual gradient handling. Decouples EP collectives from autograd scheduling entirely.
   Reusable for any MoE + AC combination.

### Recommended path

Start with fix (1) — cache router outputs through AC. It's the smallest change and
directly addresses the v154 `ScatterAddBackward0` shape error. If that works with
`use_reentrant=False`, the v153 scheduling desync is also fixed (non-reentrant AC
eliminates the interleaved FWD-recompute ordering issue).

If (1) doesn't hold under multi-step training (cache invalidation across steps), fall
back to (2). Reserve (3) for a dedicated EP sprint.

**Action**: Implement fix (1) on a held compute node with Gemma4 26B-A4B (existing EP
infrastructure). If validated, port to Qwen3-30B-A3B.

---

## Priority Order

| Priority | Task | Expected impact | Effort |
|----------|------|-----------------|--------|
| **P0** | Increase batch_size=2, grpo_samples=8 | 2-4x throughput/step | Config change |
| **P0** | Set fbs=4 (match grpo_samples) | Minor latency reduction | Config change |
| **P1** | torch.compile on expert forward | 2-5x expert compute speedup | Low |
| **P1** | IPEX FusedMoE for no_grad forwards | 2-3x policy/ref forward speedup | Medium |
| **P2** | 2-node HSDP (colocated vLLM) | Enables batch=4+, more tiles | Medium |
| **P2** | weight_sync_interval=2-4 | 13-25% throughput improvement | Config change |
| **P3** | EP fix (1): cache router through AC | Unblocks EP, realizes 11.5x compute advantage | Medium |
| **P3** | Intel Triton fused MoE kernel | Best possible expert compute | High |

---

## Success Metrics

| Metric | Current | Target (Tier 1+2) | Target (with EP) |
|--------|---------|-------------------|-------------------|
| Step time (batch=1, G=4) | 35.3s | ~30s | — |
| GRPO fwd+bwd | 23s | 15-18s | 5-8s |
| Tokens/second | ~13 | ~50-100 (batch=2-4) | ~200+ |
| Weight sync overhead | 3.3s gather + 13s reload | Same or lower | Same |
| Communication fraction | ~65% of fwd+bwd | ~50% (larger batch) | ~15% (EP reduces AG volume) |
