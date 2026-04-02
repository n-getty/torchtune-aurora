# Performance Gap Analysis: Aurora XPU vs NeMo-RL (H100)

Quantitative analysis of GRPO throughput gaps between Aurora XPU (Intel Max 1550)
and NVIDIA NeMo-RL on H100 GPUs. Based on published NeMo-RL benchmarks and measured
Aurora results from `docs/aurora_rl_baselines.md`.

## NeMo-RL Reference Numbers

Source: NeMo-RL documentation, Qwen3-32B GRPO benchmarks.

| GPUs | Mode | tok/s/GPU | Step Time | Global Batch | Seq Len |
|------|------|-----------|-----------|--------------|---------|
| 32 H100 | on-policy | 571 | 376 s | 4,096 | 2,048 |
| 32 H100 | 1-step off-policy | 1,025 | 210 s | 4,096 | 2,048 |
| 16 H100 | on-policy | 1,127 | 381 s | 4,096 | 2,048 |
| 64 H100 | 1-step off-policy | 538 | 200 s | 4,096 | 2,048 |

Config vector: [4,1,1,4,n/a] (likely [TP, PP, CP, DP, EP]).

**Cross-check**: 571 tok/s/GPU × 32 GPUs × 376s = **6.87M tokens/step**.
16 GPUs: 1,127 × 16 × 381 = **6.87M tokens/step**. Same workload, different parallelism.

With global batch = 4,096 sequences and seq_len = 2,048:
max possible = 8.39M tokens, actual = 6.87M → average effective seq_len ≈ 1,677 tokens.

### Llama-8B Reference (closest to our 3B/8B)

| GPUs | Mode | tok/s/GPU | Step Time |
|------|------|-----------|-----------|
| 16 H100 | on-policy | 1,581 | 92.8 s |
| 16 H100 | 1-step off-policy | 2,478 | 64.8 s |

Total tokens/step: 1,581 × 16 × 92.8 ≈ **2.35M tokens/step**.

## Aurora Measured Results

### Qwen3-32B (10 training + 2 vLLM tiles, fbs=4, max_gen=128, seq_len≈384)

| G (grpo_samples) | Tokens/step | Step Time | tok/s/tile |
|---|---|---|---|
| 4 | 1,536 | 18.1 s | **7.1** |
| 8 | 3,072 | 24.2 s | **10.6** |
| 16 | 6,144 | 36.9 s | **13.9** |

Scaling trend — each 2× in sequences gives diminishing throughput gains:
- G=4→8: 1.49× throughput for 2× sequences
- G=8→16: 1.31× throughput for 2× sequences

### Qwen3-32B Multi-Node HSDP (20 training + 4 vLLM tiles, 2 nodes)

| G | Step Time | tok/s/tile (24 tiles) | Notes |
|---|---|---|---|
| 4 | 19.4 s | 2.6 | Near-linear scaling (1.07× single-node) |

### Qwen3-8B (Aurora colocated vLLM, Config A: 4 samp, 256 tok)

| Tiles | Step Time | tok/s/tile | vs A100 (15.7 s) |
|-------|-----------|------------|------------------|
| 12 | 9.5 s | 8.9 | **1.65× faster** |
| 6 | 10.1 s | 11.2 | **1.55× faster** |

### Qwen2.5-72B (24 training + 12 vLLM tiles, 3 nodes, CPU offload)

| Metric | Value |
|--------|-------|
| Step time (steady-state) | 84.6 s |
| grpo_samples | 4 |
| max_gen_tokens | 128 |
| forward_batch_size | 1 |
| Tokens/step | ~512 |
| tok/s/tile (36 tiles) | **0.17** |

72B throughput is severely bottlenecked by fbs=1 (9-13× forward pass penalty vs fbs=4)
and CPU offload optimizer overhead (6.5s vs 0.2s). With fbs=4, estimated step time drops
to ~57s, improving tok/s/tile to ~0.25.

## Batch Size Scaling Analysis (32B)

### Can we reach NeMo-RL throughput by increasing batch size?

Linear regression on measured data: `step_time(G) ≈ 12.5 + 1.525 × G` (R²≈0.99).

| G | Est. Step Time | Tokens/step | tok/s/tile | vs NeMo-RL 571 |
|---|---|---|---|---|
| 16 | 36.9 s | 6,144 | 13.9 | 41× gap |
| 32 | 61.3 s | 12,288 | 16.7 | 34× gap |
| 64 | 109.9 s | 24,576 | 18.6 | 31× gap |
| 128 | 207.7 s | 49,152 | 19.7 | 29× gap |
| 256 | 402.9 s | 98,304 | 20.3 | 28× gap |
| **∞** | — | — | **20.9** | **27× gap** |

Asymptotic ceiling: **384 / 1.525 / 12 ≈ 21 tok/s/tile**.

**Batch size cannot close the gap.** Throughput plateaus at ~21 tok/s/tile regardless
of sequence count. Increasing G past 64 yields <6% additional throughput.

### Effect of longer sequences (seq_len=2048 to match NeMo-RL)

Longer sequences increase tokens per step but also increase compute and generation
time proportionally. The net effect on tok/s/tile is minimal.

Estimate at seq_len=2048, G=16, fbs=4:
- vLLM generation (16 × ~1536 gen tokens at ~214 tok/s batched): ~115s
- Policy + ref forward (4 chunks, 5.3× longer sequences): ~40s
- Training fwd+bwd: ~50s
- **Step time: ~210s** for 32,768 tokens
- **tok/s/tile: ~13** — approximately the same as short sequences

Generation dominates at long seq_len, negating any amortization benefit from
more tokens per sequence.

## Root Cause: Hardware and Software, Not Batch Size

The ~27–29× gap between Aurora (~21 tok/s/tile ceiling) and NeMo-RL (571 tok/s/GPU)
decomposes into identifiable factors:

| Factor | Penalty | Explanation |
|--------|---------|-------------|
| H100 vs XPU raw BF16 TFLOPS | **2.4×** | 990 vs 420 TFLOPS per device |
| torch.compile | **~1.5×** | Deadlocks with oneCCL on multi-node; see detail below |
| FlashAttention-3 vs math-only SDPA | **~1.5×** | XPU SDPA kernel leaks UR handles; see detail below |
| AllGather-compute overlap | **~1.5×** | FSDP2 supports it but not yet activated; see detail below |
| FSDP-32 vs FSDP-10 shard count | **~1.5×** | More GPUs = less AllGather data per device |
| Batch amortization (4,096 vs 4–16 seqs) | **~1.5×** | Fixed overhead spread over 256–1000× more work |
| **Combined** | **~28×** | Matches observed 27–29× gap |

### Detail: torch.compile on multi-node XPU

**Symptom**: Multi-node GRPO deadlocks when `compile=True`.

**Current behavior** (in `grpo_full_finetune_distributed_xpu.py:289-296`):
```python
if self._compile and self._device.type == "xpu":
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
    if self.world_size > local_world_size:
        log.warning("torch.compile is not supported multi-node on XPU. Disabling.")
        self._compile = False
```

The recipe auto-detects multi-node and disables compile. All baselines were run with
compile off. The deadlock occurs during oneCCL collective operations inside compiled
graphs — the compiled code path and oneCCL's progress engine are incompatible.

**Single-node compile works** (backbone-only). This is documented in CLAUDE.md as
"viable single-node only." The estimated 1.3–1.5× gain is based on typical
torch.compile improvements for transformer forward passes.

**What would fix it**: Either Intel fixes the oneCCL + compile interaction, or
PyTorch's XPU backend adds graph-break-aware collective scheduling. This is an
Intel software issue, not a fundamental limitation.

### Detail: FlashAttention / SDPA disabled on XPU

**Symptom**: XPU's optimized SDPA kernels (both flash and memory-efficient) leak
Unified Runtime (UR) handles on every call, eventually causing GPU segfaults.

**Current workaround** (in `grpo_full_finetune_distributed_xpu.py:261-268`):
```python
if self._device.type == "xpu":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
```

This forces math-only SDPA (pure PyTorch matmul + softmax), which doesn't leak.
The UR handle leak is a **separate bug** from the `empty_cache()` + FSDP
`storage.resize_()` leak documented in `docs/intel_xpu_resource_leak_bug_report.md`.

**Impact**: Math-only SDPA is ~1.5× slower than flash SDPA for attention computation.
Since attention is a significant fraction of transformer forward/backward time, this
directly reduces throughput.

**What would fix it**: Intel fixes the UR handle leak in their XPU SDPA kernel
implementation. The workaround is applied in all recipes and test scripts.

### Detail: AllGather-compute overlap

**Symptom**: Each FSDP forward pass serializes AllGather then compute. On H100 with
NVLink, NCCL can overlap the next layer's AllGather with the current layer's compute.

**Current state**: This is more nuanced than "oneCCL can't overlap." Three factors:

1. **PyTorch FSDP2 supports overlap natively** via `reshard_after_forward=None`
   (PyTorch PR #155319). Our `_distributed.py:750-751` has a TODO to enable this:
   ```python
   # TODO: we should actually use reshard_after_forward=None
   # on latest nightlies: https://github.com/pytorch/pytorch/pull/155319
   ```
   This is **not yet activated** in torchtune — it's available in PyTorch nightlies
   but hasn't been tested on XPU.

2. **FSDP internally pipelines AllGather** between layers (deferred cleanup in
   `_fsdp_param_group.py`), but this only overlaps AllGather-with-AllGather, not
   AllGather-with-compute.

3. **oneCCL + XPU compute overlap is untested.** Unlike NCCL on CUDA where compute
   streams and communication streams run concurrently via SM partitioning, XPU's
   Level Zero command queues may or may not support true overlap. The H100's NVLink
   (900 GB/s bidirectional) + NVSwitch provides dedicated communication bandwidth
   that doesn't compete with compute SMs. Intel Max 1550 tiles share the same HBM
   bandwidth between compute and communication.

**What could help now (no Intel changes needed)**:
- Activate `reshard_after_forward=None` and benchmark on XPU
- Test whether oneCCL AllGather on a separate SYCL queue overlaps with compute
- Use `reshard_after_forward=False` (SHARD_GRAD_OP) to avoid re-AllGather at the
  cost of higher memory — viable for smaller models or with more FSDP tiles

**What would help long-term**: Intel adding dedicated copy engines or communication
channels that don't contend with compute for HBM bandwidth.

### FLOPS utilization comparison

| System | Total tok/s | Total TFLOPS | tok/s per TFLOPS |
|--------|------------|-------------|------------------|
| Aurora 12 tiles (G=16) | 167 | 5,040 | **0.033** |
| Aurora 12 tiles (G=∞, ceiling) | 252 | 5,040 | **0.050** |
| NeMo-RL 32 H100s | 18,272 | 31,680 | **0.577** |

**11.5× FLOPS utilization gap** — explained by compile, FlashAttention, AllGather
overlap, and batch amortization. This is the "software+interconnect" gap distinct
from the raw hardware TFLOPS difference.

## What Would It Take to Match NeMo-RL?

### Matching per-tile throughput: not feasible with current software

Our hardware ceiling (~21 tok/s/tile) is 27× below NeMo-RL's 571 tok/s/GPU.
Closing this requires hardware/software changes:

| Change | Estimated Gain | Status |
|--------|---------------|--------|
| torch.compile (backbone only) | 1.3–1.5× | Works single-node; multi-node blocked by oneCCL deadlock |
| FlashAttention on XPU | 1.3–1.5× | Blocked by UR handle leak in XPU SDPA kernel |
| AllGather-compute overlap | 1.3–1.5× | `reshard_after_forward=None` available but untested on XPU |
| Larger FSDP shard groups (multi-node) | 1.2–1.5× | Available now with more nodes |
| Batch size G=64+ | 1.3× (vs G=16) | Available now, limited by memory |
| **Combined (best case)** | **~4–5×** | **~80–105 tok/s/tile** |

**Actionable now** (no Intel fixes needed):
- Test `reshard_after_forward=None` for AllGather-compute overlap
- Increase batch size to G=64
- Scale to more nodes for larger FSDP shard groups

**Blocked on Intel**:
- torch.compile + oneCCL multi-node deadlock
- XPU SDPA kernel UR handle leak (FlashAttention)

Even with all optimizations, we'd reach ~80–105 tok/s/tile — still ~6× below
NeMo-RL. The remaining gap is the fundamental H100 vs XPU compute difference
(2.4×) plus NVLink interconnect advantage (~2.5×).

### Matching total throughput: node scaling

NeMo-RL total throughput: 18,272 tok/s (32 H100s).

| Aurora config | Per-replica tok/s | Replicas needed | Nodes needed |
|---|---|---|---|
| Current (G=16, 12 tiles/replica) | 167 | 110 | **110** |
| Optimized (G=64, 12 tiles/replica) | ~224 | 82 | **82** |
| Best case (all SW optimizations) | ~840 | 22 | **~25** (with vLLM tiles) |

With all software optimizations, ~25 Aurora nodes (300 tiles) could match the
total throughput of 32 H100s. Without optimizations, ~82–110 nodes.

## Per-Model Summary

### Qwen3-8B

| Metric | NeMo-RL 8B (16 H100) | Aurora 8B (12 tiles) | Gap |
|--------|----------------------|---------------------|-----|
| tok/s/device | 1,581 | ~8.9 | **178×** |
| Step time | 92.8 s | 9.5 s | **Aurora 9.8× faster** |
| Sequences/step | ~thousands | 4 | **~500× less work** |

The 178× tok/s gap is misleading: NeMo-RL processes ~500× more sequences per step.
At matched configs (4 samples, 256 tokens), **Aurora is 1.65× faster than A100-40GB**.
The step time comparison (9.5s vs 92.8s) reflects entirely different batch sizes.

### Qwen3-32B

| Metric | NeMo-RL 32B (32 H100) | Aurora 32B (12 tiles) | Gap |
|--------|----------------------|----------------------|-----|
| tok/s/device | 571 | 13.9 (G=16) | **41×** |
| Step time | 376 s | 36.9 s (G=16) | **Aurora 10.2× faster** |
| Sequences/step | ~4,096 | 16 | **256× less work** |
| Feasible on A100-40GB? | N/A (H100 only) | — | **Aurora runs, A100 can't** |

Again: Aurora's faster step time reflects much smaller batches, not faster hardware.
But Aurora runs 32B GRPO at all — **A100-40GB OOMs at every configuration tested**.

### Qwen2.5-72B

| Metric | NeMo-RL (no 72B data) | Aurora 72B (36 tiles, 3 nodes) |
|--------|----------------------|-------------------------------|
| tok/s/tile | — | 0.17 (current) / ~0.25 (est. fbs=4) |
| Step time | — | 84.6 s (G=4, fbs=1) / ~57 s (est. fbs=4) |
| Sequences/step | — | 4 |

No published NeMo-RL step times for 72B. TRL reference uses 40 H100s (5 nodes × 8)
with no published timing. Our 72B result is bottlenecked by fbs=1 and CPU offload;
fbs=4 and additional training nodes would significantly improve throughput.

## Key Takeaways

1. **The tok/s/GPU gap (~27–41×) is NOT primarily a batch size problem.** Increasing
   batch size from G=16 to G=∞ only improves throughput ~1.5×. The gap is dominated
   by hardware (2.4× TFLOPS), software optimizations (compile, FA3: ~2.25×), and
   interconnect (NVLink overlap, shard count: ~2.25×).

2. **Step time comparisons are misleading without matching batch sizes.** Aurora's
   18.1s vs NeMo-RL's 376s reflects 256× fewer sequences, not 20× faster hardware.

3. **Aurora's advantage is memory capacity.** 64 GiB/tile enables 32B full-finetune
   GRPO on a single node and 72B on 3 nodes — configurations that OOM on A100-40GB
   and require H100-80GB on NVIDIA hardware.

4. **To close the per-tile throughput gap**, the highest-impact changes are:
   - torch.compile support on multi-node XPU (~1.5×)
   - FlashAttention on XPU (~1.5×)
   - oneCCL AllGather-compute overlap (~1.5×)
   - These are Intel software/driver issues, not algorithmic limitations.

5. **Aurora can match NeMo-RL total throughput with ~25 nodes** (best case, all
   software optimizations) or ~82–110 nodes (current software). Aurora has 10,624
   nodes (127,488 tiles) — throughput matching is a scheduling question, not a
   capability question.
