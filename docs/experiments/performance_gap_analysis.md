# Performance Gap Analysis: Aurora XPU vs NeMo-RL (H100)

Quantitative analysis of GRPO throughput gaps between Aurora XPU (Intel Max 1550)
and NVIDIA NeMo-RL on H100 GPUs. Based on published NeMo-RL benchmarks and measured
Aurora results from `docs/experiments/aurora_rl_baselines.md`.

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

| Factor | Estimated Penalty | Measured (2026-04-02) | Explanation |
|--------|---------|---------|-------------|
| H100 vs XPU raw BF16 TFLOPS | **2.4×** | — | 990 vs 420 TFLOPS per device |
| torch.compile | **~1.5×** | **~1.0× (unstable)** | Recompiles on variable shapes; net zero or negative |
| FlashAttention-3 vs math-only SDPA | **~1.5×** | **1.0×** | No difference — XPU math-only already optimized |
| AllGather-compute overlap | **~1.5×** | **1.0× (single-node)** | No benefit on 10 tiles; may help multi-node |
| FSDP-32 vs FSDP-10 shard count | **~1.5×** | — | More GPUs = less AllGather data per device |
| Batch amortization (4,096 vs 4–16 seqs) | **~1.5×** | — | Fixed overhead spread over 256–1000× more work |
| **Combined (estimated)** | **~28×** | **~9×** | Software gaps mostly don't exist in practice |

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

**IMPORTANT: Needs retesting.** The multi-node compile deadlock was observed when
our CCL configuration was severely broken (`CCL_WORKER_COUNT=4` causing 48× AllGather
regression, `CCL_REDUCE_SCATTER=ring` causing 63× ReduceScatter regression, ofi
transport instead of MPI). These CCL issues caused collectives to run 10–100× slower
and created massive thread contention (40 CCL worker threads competing for Level Zero
device access). The "compile deadlock" may have been a symptom of this contention
rather than a fundamental compile + oneCCL incompatibility.

Timeline:
- Compile deadlock observed: 2026-03-30 (CCL still broken)
- CCL root causes fixed: 2026-04-01 (WORKER_COUNT=1, MPI transport, no RS=ring)
- **Compile has NOT been retested with fixed CCL configuration**

**Test results (2026-04-02)**:

*Static compile (compile=True, dynamic=False)*: No deadlock, but **recompilation on
every shape change**. Forward alternates between 1.5s (cached) and 9.7s (recompiling)
due to variable sequence lengths. Steps 0-2 warmup (87.5, 68.2, 27.8s). Step 3 hit
17.8s (cached shapes), step 4 regressed to 27.5s.

*Dynamic compile (compile=True, compile_dynamic=True)*: Uses symbolic shapes (same
approach as PRISM's `torch.compile(model.backbone, dynamic=True)`). Eliminates
repeated graph tracing, but XPU inductor backend's Triton-to-SYCL kernel compilation
takes **~40s per unique shape**. Forward: 211s (step 0 warmup), 40-41s (steps 1-2
recompile), 1.5s (step 3 cache hit), 40.9s (step 4 recompile). Net result: **worse
than static compile** because dynamic kernel compilation overhead exceeds static's
re-tracing cost.

*foreach optimizer*: `optimizer.foreach=True` tested alongside baseline. No change
(0.2s → 0.2s). Optimizer is 1% of step time; FSDP sharding already makes per-param
updates small.

**Root cause**: The XPU inductor backend compiles Triton kernels via SYCL/Level Zero,
which is fundamentally slow (~40s per unique kernel). On CUDA, Triton compiles to
PTX in <1s. NeMo-RL and TRL both avoid compile entirely for RL training — NeMo-RL
uses sequence packing instead, TRL uses eager mode. Neither has solved compile +
variable-length RL on any hardware.

**Viable paths forward**:
1. **Sequence packing** (NeMo-RL approach): bin-pack variable-length responses into
   fixed-capacity bins, reducing padding waste without compile. Requires FlashAttention
   or equivalent for correct packed attention masking.
2. **Fixed-length padding**: pad all sequences to `max_seq_len`, eliminating shape
   variation. Wastes compute on padding but makes compile stable. Only viable if
   padding overhead < compile gain.
3. **Wait for faster XPU inductor**: Intel's Triton-to-SYCL compilation may improve
   in future PyTorch/oneAPI releases.

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

**However: the evidence for a separate SDPA leak is weak.** The code comment says
"XPU SDPA kernel leaks UR handles" but:

1. The `intel_xpu_resource_leak_bug_report.md` lists "Math-only SDPA" under "Ruled
   Out" with "No effect" — meaning switching SDPA mode didn't fix the `empty_cache()`
   leak (the only UR leak we've isolated).
2. There is **no independent test** isolating an SDPA-specific UR leak. The SDPA
   disable was added as a "separate precaution" (CLAUDE.md) during the same period
   when CCL was severely misconfigured.
3. The CCL misconfiguration (`CCL_WORKER_COUNT=4`) created 40 worker threads
   competing for Level Zero device access. This thread contention may have
   exacerbated or entirely caused UR handle pressure that was attributed to SDPA.

**Timeline:**
- SDPA disable added: 2026-03-30 (CCL still broken, WORKER_COUNT=4)
- CCL root causes fixed: 2026-04-01 (WORKER_COUNT=1, MPI transport)
- **SDPA has NOT been retested with fixed CCL configuration**
- **SDPA has NOT been retested after removing `empty_cache()` calls** (the only
  proven UR leak trigger)

**Impact if real**: Math-only SDPA is ~1.5× slower than flash SDPA for attention.

**Test result (2026-04-02)**: SDPA tested with flash+mem_efficient enabled, fixed
CCL, no `empty_cache()`. Ran 5 steps with zero UR errors or crashes. Step times
identical to math-only baseline (avg 18.3s vs 18.2s). **The SDPA UR leak hypothesis
is disproven** — the leak was from `empty_cache()`, not SDPA. However, re-enabling
optimized SDPA provides **no throughput improvement**. Safe to enable but pointless
for performance.

#### Why Flash Attention provides no throughput gain on XPU

**Intel's FA implementation**: Intel's flash attention on XPU uses **SYCL-TLA**
(a fork of NVIDIA's CUTLASS adapted for SYCL), not oneDNN. PyTorch wraps it via
`aten/src/ATen/native/transformers/xpu/attention.cpp` calling
`sycltla::flash_attention_forward()`. It implements Flash Attention V2 (tiled
online softmax, O(N) memory) like Tri Dao's CUDA version, but with key
architectural differences:

| Aspect | CUDA (Tri Dao) | XPU (SYCL-TLA) |
|--------|---------------|-----------------|
| Matrix ops | Tensor cores (WMMA/MMA) | DPAS (Dot Product Accumulate Systolic) |
| Alignment | 8-byte on head dim | **64-byte** on head dim (adds pad/unpad overhead) |
| Execution | CUDA warps (32 threads) | SYCL subgroups (16-32 work items) |
| K layout | Row-major | **Column-major** (transposed) |
| Maturity | ~95% of peak | ~78% of peak (SYCL-TLA v0.7) |

There is **no memory-efficient attention backend** for XPU — only flash or math.
Setting `enable_mem_efficient_sdp(True)` is accepted but never dispatches.

**Four reasons FA doesn't help in GRPO training on PVC:**

1. **Attention is not the bottleneck.** GRPO step time is dominated by generation
   (autoregressive decoding), AllGather communication, and reward computation.
   Attention is a small fraction of total compute. Even a 2× attention speedup
   would be invisible in end-to-end step time.

2. **PVC's HBM bandwidth reduces FA's advantage.** Flash attention's core benefit
   is reducing HBM traffic via tiling. PVC has **3.2 TB/s aggregate HBM bandwidth**
   (across 2 stacks) vs A100's 2.0 TB/s. Higher bandwidth means math-only SDPA
   (which makes more HBM round-trips) is less severely bottlenecked — the hardware
   compensates for the algorithmic inefficiency.

3. **Math-only SDPA on XPU is already well-optimized.** The math path dispatches
   through `torch.matmul` → oneDNN GEMM kernels, which are highly tuned for PVC's
   systolic arrays. The SYCL-TLA FA kernel is still maturing (~78% of peak) and
   carries 64-byte alignment overhead that the math path avoids.

4. **RL sequence lengths are moderate.** FA's advantage grows with sequence length
   (O(N²) → O(N) memory). At typical RL lengths (512–2048 tokens), the memory
   savings don't translate to meaningful compute savings. FA becomes decisive at
   4K+ tokens where quadratic memory pressure dominates.

**When FA would matter on XPU:**
- Sequence packing with 4K+ contexts (need FlashAttention + BlockMask for correct
  packed attention masking to avoid cross-contamination between packed samples)
- Workloads where generation is removed (e.g., pre-generated via vLLM), making
  the training step more attention-heavy
- Future SYCL-TLA versions closing the gap to peak performance

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

**Test result (2026-04-02)**: `reshard_after_forward=None` tested on single node
(10 tiles). No crashes. Step times identical to baseline (avg 18.3s vs 18.2s).
AllGather within a single node is fast enough via shared memory/PCIe that
overlapping provides no benefit. **May still help multi-node** where AllGather
crosses Slingshot fabric — worth retesting with 2+ node HSDP if multi-node compile
becomes viable.

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
| torch.compile (backbone only) | 1.3–1.5× | **Retest needed** — deadlock may have been caused by broken CCL |
| FlashAttention / SDPA on XPU | 1.3–1.5× | **Retest needed** — no isolated evidence of SDPA-specific leak |
| AllGather-compute overlap | 1.3–1.5× | `reshard_after_forward=None` available but untested on XPU |
| Larger FSDP shard groups (multi-node) | 1.2–1.5× | Available now with more nodes |
| Batch size G=64+ | 1.3× (vs G=16) | Available now, limited by memory |
| **Combined (best case)** | **~4–5×** | **~80–105 tok/s/tile** |

**Optimization test results (2026-04-02):**

All three optimizations were tested on a single node (10 training + 2 vLLM tiles,
Qwen3-32B, G=4, fbs=4, max_gen=128) with the fixed CCL configuration
(WORKER_COUNT=1, MPI transport). Each test ran 5 steps. Baseline: 17.7–18.7s/step
(avg 18.2s).

| Optimization | Result | Step Time | Notes |
|---|---|---|---|
| SDPA (flash+mem_efficient enabled) | **Safe, no improvement** | 18.3s avg | No UR errors, no crashes. Identical to baseline. |
| torch.compile (single-node) | **Unstable** | 17.8–87.5s | Recompilation on variable-length inputs. Forward alternates 1.5s/9.7s. |
| AllGather-compute overlap | **Safe, no improvement** | 18.3s avg | `reshard_after_forward=None` works but no measurable gain on 10 tiles. |

**Key findings:**

1. **SDPA**: The flash+mem_efficient SDPA kernels are **not leaking UR handles** —
   the original leak was caused by `empty_cache()` (already removed). Safe to
   re-enable, but provides **no throughput improvement** on XPU. The math-only
   fallback performs identically. This means the ~1.5× penalty attributed to
   math-only SDPA was incorrect — XPU's math-only SDPA is already well-optimized,
   or the optimized kernels aren't actually being used despite being "enabled."

2. **torch.compile**: Works on single-node but **recompiles on every shape change**.
   GRPO generates variable-length sequences, so tensor shapes change every step.
   This causes the compiled forward to alternate between cached (1.5s) and
   recompiling (9.7s) — net negative vs baseline. To benefit from compile, would
   need either (a) fixed-length padding of all sequences or (b) `torch.compile`
   with `dynamic=True` shapes support on XPU. Neither is currently viable.

3. **AllGather-compute overlap**: `reshard_after_forward=None` is accepted by FSDP2
   without errors, but provides zero measurable improvement on single-node (10
   tiles). The AllGather within a single node is fast enough (~ms via shared memory
   / PCIe) that overlapping it with compute provides no benefit. **May still help
   on multi-node** where AllGather crosses the Slingshot fabric — worth retesting
   with 2+ node HSDP.

### Batch Size Scaling Tests (2026-04-02)

G=16 is the practical max for 10-tile single-node FSDP. G=32 OOMs.

| G | Step Time | grpo phase | Tokens/step | tok/s/tile | Status |
|---|-----------|-----------|------------|-----------|--------|
| 4 | 18.2s | 5.9s | 1,536 | 7.1 | Stable (5+ steps) |
| 8 | 24.2s | — | 3,072 | 10.6 | Stable |
| 16 | 36.9s | — | 6,144 | 13.9 | Stable |
| 32 | 69.5s (step 0) | 30.4s | 12,288 | — | **OOM step 1 backward** |

### Chunked Cross-Entropy Loss (2026-04-02)

`GRPOWithChunkedOutputLoss` (8 chunks) processes logits in chunks to reduce peak
logit-tensor memory. At G=4 with vocab=151,936: logit tensor is
`[4, 128, 151936]` × 4 bytes = ~300 MB in fp32. Chunking reduces this to ~37 MB
per chunk. The savings scale with G: at G=32, the logit tensor is ~2.4 GB.

**G=4 with chunked loss vs baseline (same node, same config, same seed)**:

| Metric | Baseline (GRPOSimpleLoss) | Chunked (8 chunks) | Difference |
|--------|--------------------------|-------------------|------------|
| Step 0 | 23.9s (grpo=7.5s) | 25.2s (grpo=8.8s) | +1.3s (+5%) |
| Step 1 | 18.2s (grpo=5.9s) | 18.2s (grpo=5.8s) | identical |
| Step 2 | 18.7s (grpo=6.3s) | 18.5s (grpo=6.1s) | identical |
| Step 3 | 17.7s (grpo=6.1s) | 17.5s (grpo=6.0s) | identical |
| Step 4 | 18.2s (grpo=6.0s) | 18.3s (grpo=5.9s) | identical |
| kl_loss (step 1) | 0.000630 | 0.000541 | Different seed state |
| rewards | Match | Match | Same reward patterns |

**Conclusion at G=4**: No throughput difference. Chunked loss is a zero-cost
optimization at small batch sizes where the logit tensor is already small.

**G=32 with chunked loss** (test whether reducing logit memory enables G=32):

| Step | Step Time | grpo forward | backward | Result |
|------|-----------|-------------|---------|--------|
| 0 | 69.5s | 8.5s | 21.9s | OK |
| 1 | — | 7.7–37.8s (ranks vary) | **OOM** | Crash |

G=32 plain (GRPOSimpleLoss) OOMed on the **forward** (SDPA) at step 2.
G=32 chunked survived the forward but OOMed on the **backward** at step 1.
The chunked loss shifted the OOM failure point from forward to backward by
reducing peak logit memory, but the model forward/backward itself (SDPA
attention + activation checkpointing recompute) is the memory bottleneck at
G=32, not logit materialization.

**Memory impact**: The recipe logs peak memory only at model init (7.69 GiB
policy, 13.93 GiB policy+ref), not per-step. We do not have measured per-step
peak memory data. The theoretical memory savings from chunked loss at G=32 are:
- Logit tensor at G=32: `[32, 128, 151936]` × 4B = **2.35 GiB** (fp32)
- Chunked (8 chunks): `[32, 16, 151936]` × 4B = **0.29 GiB** per chunk
- **Saving: ~2.06 GiB** — significant but insufficient when OOM is in SDPA

**When chunked loss matters**: For CoT or longer-generation configs where
`max_generated_tokens >> 128`, the logit tensor grows proportionally and
chunked loss becomes critical for memory. At max_gen=2048 with G=16:
`[16, 2048, 151936]` × 4B = **19 GiB** — chunked loss reduces this to
~2.4 GiB per chunk.

### Sequence Packing Analysis (2026-04-02)

Sequence packing (NeMo-RL's primary optimization) bin-packs variable-length
sequences into fixed-capacity packs with block-diagonal attention masks. This
eliminates padding waste in the training forward/backward pass.

**Implementation**: Created `torchtune/dev/grpo/packing.py` with greedy
first-fit-decreasing bin packing, block-diagonal mask creation, and
autograd-compatible unpack. Unit tests pass including gradient flow verification.

**Result for our config (context=384, max_gen=128, total_len=512)**:

Each sequence has actual length 414–512 tokens (81–100% of total_len). With
minimum sequence length 414/512 = 81%, no two sequences can fit in one pack
(414 + 414 = 828 > 512). All 16 sequences get their own pack. **0% padding
reduction.**

| Config | Min fill | Max fill | Packs (from 16 seqs) | Reduction |
|--------|----------|----------|---------------------|-----------|
| ctx=384, gen=128, total=512 | 81% | 100% | 16 | **0%** |
| ctx=128, gen=512, total=640 | 28% | 100% | 11 | **31%** |
| ctx=64, gen=2048, total=2112 | 8% | 100% | 10 | **38%** |

**Conclusion**: Packing is only effective when `max_generated_tokens >>
context_length`, which creates high variance in sequence lengths and low fill
rates. For our GSM8K config (short responses, long prompts), packing provides
zero benefit. For CoT/reasoning tasks with long responses and short prompts,
packing can reduce training compute by 30-40%.

**Revised gap analysis**: The ~3.4–4.5× "software optimization gap" does not
materialize. The actual per-tile throughput gap is dominated by:
- Hardware: 2.4× raw TFLOPS difference (H100 990 vs XPU 420)
- NVLink interconnect: ~2.5× bandwidth advantage for communication overlap
- Batch amortization: ~1.5× (4,096 vs 4-16 sequences)
- **Total: ~9× (down from prior 28× estimate)**

The prior 28× estimate over-counted by assuming SDPA, compile, and overlap penalties
that don't actually exist in practice. The remaining ~9× gap (vs the 2.4× hardware
floor) is primarily interconnect and batch amortization — addressable by scaling
nodes, not by software changes.

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

## H100 NVL Measured Baselines (2026-04-02)

Direct measurements on lambda13 (8× NVIDIA H100 NVL, 94 GiB each, NVLink).
Framework: TRL 1.0.0 + vLLM 0.17.1 + DeepSpeed ZeRO-3. Matched Aurora
hyperparameters: beta=0.01, max_gen=128, BF16, gradient checkpointing.

### Qwen3-32B on H100 NVL

**Architecture**: 8 training GPUs (DeepSpeed ZeRO-3, CPU offload for optimizer)
+ colocated vLLM (TP=2, sleep mode, gpu_mem=0.35). All 8 GPUs participate in
both training and generation via vLLM sleep/wake.

| G | Step Time (steady) | All Steps Avg | Peak Mem/GPU | Notes |
|---|---|---|---|---|
| 4 | **58.2s** | 58.7s | 52.3 GiB | Primary comparison |
| 8 | **58.5s** | 59.0s | — | Near-identical to G=4 |
| 16 | **44.5s** | 46.3s | — | Alternating gen/train pattern |

G=16 is faster than G=4 because TRL batches generation into larger groups
(generation_batch_size=16), amortizing vLLM startup/teardown. Steps alternate
between ~60s (generation) and ~27s (training only).

**CPU offload penalty**: Without CPU offload, 8 GPUs OOM on optimizer step
(needs ~48 GiB/GPU for policy+ref+optimizer). CPU offload moves Adam states
to system RAM, adding ~10-15s overhead per step.

**6-GPU server mode** (for reference): 6 training GPUs + 2 dedicated vLLM GPUs
= **87.3s/step** avg. Slower due to fewer training shards AND alternating
generation/training step pattern.

### Qwen2.5-72B on H100 NVL

**Architecture**: 4 training GPUs (DeepSpeed ZeRO-3, CPU offload) on GPUs 0-3
+ 4 dedicated vLLM server GPUs (TP=4, gpu_mem=0.85) on GPUs 4-7.

| G | Step Time (steady) | All Steps Avg | Peak Mem/GPU | Notes |
|---|---|---|---|---|
| 4 | **150.8s** | 154.4s | 75.8 GiB (train) | 4+4 split |

72B could not run in colocated mode (vLLM + training compete for GPU memory).
Server mode with 4 training + 4 generation GPUs was the only viable config
on a single 8-GPU node.

### Aurora vs H100 NVL Comparison (Matched Configs)

#### Same-Code, Same-Architecture Comparison: torchtune + vLLM on Both Platforms

The fairest comparison uses the **same torchtune GRPO recipe AND same vLLM server
mode architecture** on both platforms, eliminating all framework AND generation
method differences.

| Model | Config | Aurora XPU | H100 NVL | **Aurora Speedup** |
|-------|--------|-----------|----------|-------------------|
| Qwen3-32B | G=4, 128tok, vLLM gen | 18.1s (10+2 tiles, FSDP-10, vLLM TP=2) | **15.3s** (6+2 GPUs, FSDP-6, vLLM TP=2, CPU offload) | **0.85×** (H100 faster) |
| Qwen3-32B | G=4, 128tok, native gen | 18.1s (10+2 tiles, FSDP-10, vLLM gen) | **24.7s** (8 GPUs, FSDP-8, native gen, CPU offload) | **1.36×** |

**Fair comparison (vLLM on both): H100 NVL is ~18% faster than Aurora.**

H100 vLLM details: `benchmarks/qwen32B_grpo_h100_vllm.yaml`, FSDP-6 (GPUs 0-5)
with CPU offload, vLLM server on GPUs 6-7 (TP=2), BF16, activation checkpointing.
All ranks call vLLM concurrently (3.2-3.3s generation, ~160 tok/s). Peak memory:
~68 GiB/GPU (step 1), ~67 GiB (steady). Per-step range: 15-16s (steps 2-5, from
FSDP-6 run before OOM due to GPU sharing with another user). Confirmed stable at
20.9s/step with FSDP-3 (3 GPUs) over 12 steps.

**Why the H100 is faster in the fair comparison:**
- **H100 NVL TFLOPS**: 990 TFLOPS BF16 (tensor cores) vs Intel Max 1550: 410 TFLOPS BF16
  → 2.4× raw compute advantage
- **FlashAttention-2**: H100 has optimized FA2; XPU uses math-only SDPA
- **NVLink bandwidth**: 900 GB/s (H100 NVL) vs Xe Link: 128 GB/s → faster FSDP
  all-gathers and all-reduces
- **vLLM generation is ~equal**: H100 3.2s vs Aurora ~3.5s (small difference)
- **Net effect**: H100's 2.4× compute advantage is partially offset by Aurora's
  1.67× more training tiles (10 vs 6), resulting in ~18% H100 advantage

**Previous native gen comparison (1.36× Aurora advantage) was unfair** because
Aurora used vLLM for generation while H100 used slow native PyTorch autoregressive
generation. With vLLM on both, the generation bottleneck is equalized.

**Architectural comparison:**
| | Aurora | H100 NVL |
|---|---|---|
| Training tiles/GPUs | 10 tiles | 6 GPUs |
| vLLM tiles/GPUs | 2 tiles (TP=2) | 2 GPUs (TP=2) |
| Model shard size | 32B/10 = 3.2B = 6.4 GiB | 32B/6 = 5.3B = 10.7 GiB |
| Memory per accelerator | 64 GiB | 94 GiB |
| CPU offload needed | No | Yes (optimizer states) |
| Total train memory | 640 GiB | 564 GiB |

#### Cross-Framework Comparison (for reference)

| Model | Config | Aurora (torchtune) | H100 NVL (other frameworks) | Speedup |
|-------|--------|-------------------|---------------------------|---------|
| Qwen3-32B | G=4, 128tok | 18.1s | 58.2s (TRL+DS3+vLLM) | 3.2× |
| Qwen3-32B | G=4, 128tok | 18.1s | 76.5s (verl+FSDP+vLLM, BF16) | 4.2× |
| Qwen3-32B | G=4, 128tok | 18.1s | **15.3s (torchtune+vLLM, same code+arch)** | **0.85× (H100 faster)** |
| Qwen3-32B | G=4, 128tok | 18.1s | 24.7s (torchtune, native gen) | 1.36× |
| Qwen3-32B | G=8, 128tok | 24.2s | 58.5s (TRL+DS3+vLLM) | 2.4× |
| Qwen3-32B | G=16, 128tok | 36.9s | 44.5s (TRL+DS3+vLLM) | 1.2× |
| Qwen2.5-72B | G=4, 128tok | 84.6s (36 tiles, 3 nodes) | 150.8s (TRL, 4+4 GPUs) | 1.8× |

The TRL/verl H100 numbers are **not fair** comparisons — they are bottlenecked
by framework overhead (Accelerate, DeepSpeed ZeRO-3 instead of FSDP2, CPU offload
policy, vLLM reshard overhead). The same-code torchtune comparison (1.36×) is the
correct baseline.

**Why TRL/verl are slower than torchtune on the same H100 hardware:**
- TRL: Forced into DeepSpeed ZeRO-3 (FSDP1 bug), Accelerate wrapper overhead,
  vLLM sleep/wake transition cost
- verl: FSDP reshard overhead (6.3s/step = 8%), actor model FP32→BF16 conversion
  issues, Hydra config complexity, Ray orchestration overhead

## Key Takeaways

1. **Fair same-code+architecture comparison: H100 NVL is ~18% faster than Aurora.**
   Using the identical torchtune GRPO recipe AND same vLLM server mode architecture
   on both platforms (G=4, max_gen=128, beta=0.01), H100's 6+2 GPUs deliver 15.3s/step
   vs Aurora's 10+2 tiles at 18.1s/step. H100's 2.4× raw TFLOPS advantage is partially
   offset by Aurora's 1.67× more training accelerators (10 vs 6 tiles).

2. **Framework choice dominates hardware performance for small-batch RL.** The same
   H100 hardware shows 3.8× variance across frameworks:
   - torchtune + vLLM (FSDP2, same arch as Aurora): **15.3s/step**
   - torchtune native gen (FSDP2, no vLLM): **24.7s/step**
   - TRL (Accelerate + DeepSpeed ZeRO-3): **58.2s/step**
   - verl (FSDP + Ray + vLLM): **76.5s/step**

   This means the framework penalty (58.2/15.3 = 3.8×) is **much larger** than
   the hardware difference (18.1/15.3 = 1.18×). The earlier 3.2× comparison
   (Aurora torchtune vs H100 TRL) conflated framework overhead with hardware.

3. **The tok/s/GPU gap (~27–41× vs NeMo-RL) is NOT primarily a batch size problem.**
   Increasing batch size from G=16 to G=∞ only improves throughput ~1.5×. The gap
   is dominated by hardware (2.4× TFLOPS), software optimizations (compile, FA3:
   ~2.25×), and interconnect (NVLink overlap, shard count: ~2.25×).

4. **Step time comparisons are misleading without matching batch sizes.** Aurora's
   18.1s vs NeMo-RL's 376s reflects 256× fewer sequences, not 20× faster hardware.

5. **Aurora's advantage is memory capacity + native PyTorch.** 64 GiB/tile enables
   32B full-finetune GRPO on a single node without CPU offload — H100 NVL (94 GiB)
   requires CPU offload even with 8 GPUs. Combined with torchtune's zero-overhead
   distributed stack, this gives Aurora a step-time advantage despite lower raw TFLOPS.

6. **The "software optimization gap" was overestimated.** Comprehensive testing
   (2026-04-02) of seven optimizations showed none improve throughput:

   | Optimization | Result |
   |---|---|
   | SDPA flash+mem_efficient | No effect (math-only already fast) |
   | torch.compile (static) | Unstable (recompiles on variable shapes) |
   | torch.compile (dynamic) | Worse (40s/kernel XPU inductor) |
   | AllGather-compute overlap | No effect single-node |
   | foreach optimizer | No effect (optimizer is 1% of step) |
   | Chunked cross-entropy | No throughput change; saves memory at high G |
   | Sequence packing | 0% reduction (sequences 81%+ full) |

   The actual gap vs NeMo-RL is ~9×, not ~28× — dominated by hardware TFLOPS (2.4×),
   interconnect (2.5×), and batch amortization (1.5×).

7. **G=16 is the maximum batch size** for 10-tile FSDP on 64 GiB tiles with
   32B models. G=32 OOMs even with chunked loss.

8. **H100 CPU offload is required for 32B GRPO.** On 8× H100 NVL (94 GiB each),
   32B policy + 32B ref + optimizer states = ~96 GiB per GPU before activations.
   CPU offload for optimizer frees ~16 GiB, bringing steady-state to 67 GiB.
   Aurora's FSDP-10 (10 shards vs 8) reduces per-shard memory enough to avoid
   offload entirely.
