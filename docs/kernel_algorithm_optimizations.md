# Kernel & Algorithm Optimizations: XPU Attention and Beyond

Analysis of Flash Attention, SDPA backends, and other kernel-level optimizations
for Intel XPU (Data Center GPU Max 1550 / Ponte Vecchio) on Aurora.

## Table of Contents

- [Flash Attention 2 on XPU](#flash-attention-2-on-xpu)
- [Flash Attention 3: Not Feasible on PVC](#flash-attention-3-not-feasible-on-pvc)
- [Attention Compute as Fraction of Total FLOPs](#attention-compute-as-fraction-of-total-flops)
- [When Flash Attention Actually Matters](#when-flash-attention-actually-matters)
- [Other Kernel Optimizations](#other-kernel-optimizations)
- [Test Log](#test-log)

---

## Flash Attention 2 on XPU

### Implementation

Intel's FA2 on XPU uses **SYCL-TLA** (a fork of NVIDIA's CUTLASS adapted for SYCL),
not oneDNN. PyTorch wraps it via `aten/src/ATen/native/transformers/xpu/attention.cpp`
calling `sycltla::flash_attention_forward()`. It implements Flash Attention V2 (tiled
online softmax, O(N) memory) like Tri Dao's CUDA version.

Key architectural differences from CUDA FA2:

| Aspect | CUDA (Tri Dao) | XPU (SYCL-TLA) |
|--------|---------------|-----------------|
| Matrix ops | Tensor cores (WMMA/MMA) | DPAS (Dot Product Accumulate Systolic) |
| Head dim alignment | 8-byte | **64-byte** (adds pad/unpad overhead) |
| Execution model | CUDA warps (32 threads) | SYCL subgroups (16-32 work items) |
| K layout | Row-major | **Column-major** (transposed) |
| Kernel maturity | ~95% of peak | ~78% of peak (SYCL-TLA v0.7) |

There is **no memory-efficient attention backend** for XPU — only flash or math.
Setting `enable_mem_efficient_sdp(True)` is accepted but never dispatches on XPU.

### XPU Constraints (from PyTorch sdp_utils.cpp)

- **Hardware**: Only PVC, PVC-VG, and BMG-G21 (Battlemage)
- **Data types**: bf16 or fp16 only (no fp32 input)
- **Head dimension**: Maximum 192, must match across Q/K/V
- **Causal mask**: Only when `seqlen_q == seqlen_k`
- **No deterministic mode**: Rejected when deterministic algorithms enabled
- **No nested tensors**

When any constraint fails, PyTorch falls back to math-only SDPA.

### Test Results

**2026-04-02**: Tested FA2 (`force_math_sdpa=False`) with fixed CCL config and no
`empty_cache()`. Ran 5 steps with zero UR errors or crashes.

| Backend | Avg Step Time | Notes |
|---------|---------------|-------|
| Math-only SDPA | 18.2 s/step | Baseline (32B, 10 tiles, G=4) |
| Flash + mem_efficient | 18.3 s/step | No improvement |

**Conclusion**: Safe to enable, but no throughput improvement. See analysis below.

### Why FA2 Shows No Improvement on XPU

**1. Attention scores are a tiny fraction of total compute.**
At typical GRPO sequence lengths (512-2048 tokens), attention score computation
(the part FA optimizes: `softmax(QK^T/sqrt(d))V`) is only **2-4% of total FLOPs**.
The other 96-98% is linear projections (Q/K/V/O, gate/up/down) — pure GEMMs that
FA doesn't touch. See [detailed breakdown below](#attention-compute-as-fraction-of-total-flops).

**2. PVC's HBM bandwidth reduces FA's advantage.**
FA's core benefit is reducing HBM traffic via tiling. PVC has **3.2 TB/s aggregate
HBM bandwidth** (across 2 stacks) vs A100's 2.0 TB/s. Higher bandwidth means
math-only SDPA (which makes more HBM round-trips) is less bottlenecked — the
hardware compensates for the algorithmic inefficiency.

**3. Math-only SDPA on XPU is already well-optimized.**
The math path dispatches through `torch.matmul` → oneDNN GEMM kernels, which are
highly tuned for PVC's systolic arrays. The SYCL-TLA FA kernel is still maturing
(~78% of peak) and carries 64-byte alignment overhead that the math path avoids.

**4. RL sequence lengths are moderate.**
FA's advantage grows with sequence length (O(N²) → O(N) memory). At 512-2048
tokens, the memory savings don't translate to meaningful compute savings. FA becomes
significant at 4K+ tokens.

### Configuration

In `grpo_full_finetune_distributed_xpu.py`:
```python
# force_math_sdpa: True (default) — disables flash/mem_efficient SDPA
# force_math_sdpa: False — enables optimized backends (safe but no perf gain)
if self._device.type == "xpu" and cfg.get("force_math_sdpa", True):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
```

Note: `torch.backends.cuda.enable_flash_sdp()` controls XPU FA too, despite the
`cuda` namespace — this is how PyTorch routes backend selection for all accelerators.

---

## Flash Attention 3: Not Feasible on PVC

FA3 (Shah et al., July 2024) pushes H100 utilization from **35% (FA2) to 75%
(740 TFLOPS FP16)** via three Hopper-specific innovations:

### FA3 Innovations and Hardware Dependencies

**1. Warp Specialization (Producer-Consumer Asynchrony)**
Producer warps handle async memory loads via H100's **TMA** (Tensor Memory
Accelerator) while consumer warps compute GEMMs via **WGMMA** (async tensor
core). Fully overlaps memory latency with compute.

- PVC equivalent: **None.** No dedicated DMA engine; EU threads must participate
  in all loads. No `setmaxnreg` for dynamic register reallocation between
  thread roles.

**2. Pingpong Scheduling (Hiding Softmax Under GEMM)**
Two warp groups alternate: one does GEMM, the other does softmax. Hides the
**256× throughput gap** between matmul (989 TFLOPS) and exp (3.9 TFLOPS) on H100.

- PVC equivalent: **Partial.** Concept is portable but requires async GEMM
  (WGMMA) to work. PVC's DPAS is synchronous — softmax and DPAS must
  serialize, leaving the systolic array idle during softmax.

**3. FP8 with Block Quantization**
FP8 doubles tensor core throughput to ~1.2 PFLOPS. Block quantization (one scale
per tile) fuses at zero cost since FA already tiles.

- PVC equivalent: **None.** PVC XMX supports BF16/FP16/INT8 only. FP8 is a
  Gaudi 3 / Falcon Shores feature.

### Hardware Comparison

| H100 Feature | PVC Equivalent | Gap |
|---|---|---|
| TMA (async DMA) | Software prefetch, tensor descriptors | **Significant** |
| WGMMA (async tensor core) | DPAS (synchronous systolic) | **Significant** |
| Warp specialization | No equivalent | **Significant** |
| FP8 tensor cores | Not available on PVC | **Complete gap** |
| 228 KB SMEM/SM | 512 KB SLM/Xe core | PVC advantage |
| 50 MB L2 | ~408 MB L2 (rambo caches) | PVC advantage |

### Portable FA3 Ideas

Two FA3 techniques are pure math and could be applied on XPU:
- **Block quantization**: one scale per tile of Q/K/V — useful if FP8 XMX becomes
  available on future Intel GPUs (Falcon Shores)
- **Incoherent processing**: multiply Q,K by random orthogonal Hadamard matrix M
  before quantization. Since `(QM)(KM)^T = QK^T`, output is unchanged but outliers
  are spread out, reducing quantization error by 2.6×

### Best Realistic Path: Optimized FA2 via Intel Triton

The Intel Triton backend has a working FA2 with:
- Tensor descriptors (`tl.make_tensor_descriptor`) for optimized DPAS feeding
- Multi-stage software pipelining (2-4 stages)
- Autotuning across block sizes, warp counts, GRF modes

Realistic ceiling: **40-55% of DPAS peak** (~340-460 TFLOPS BF16), vs FA3's 75%
on H100. The gap is structural — without async GEMM, softmax-DPAS serialization
is unavoidable.

---

## Attention Compute as Fraction of Total FLOPs

### Architecture Reference (Qwen2.5 family, all use GQA + SwiGLU FFN)

| Parameter | 7B | 32B | 72B |
|---|---|---|---|
| Layers | 28 | 64 | 80 |
| Hidden dim (d) | 3584 | 5120 | 8192 |
| Intermediate dim (d_ff) | 18944 | 27648 | 29568 |
| Q heads | 28 | 40 | 64 |
| KV heads | 4 | 8 | 8 |
| GQA ratio | 7:1 | 5:1 | 8:1 |
| Head dim | 128 | 128 | 128 |
| d_ff / d ratio | 5.29× | 5.40× | 3.61× |

### Per-Layer FLOP Decomposition (Forward Pass)

For sequence length `s`, hidden dim `d`, KV dim `d_kv = h_kv × d_h`:

| Component | FLOPs | What optimizes it |
|---|---|---|
| Q projection | `2·s·d²` | oneDNN GEMM |
| K projection | `2·s·d·d_kv` | oneDNN GEMM |
| V projection | `2·s·d·d_kv` | oneDNN GEMM |
| **Attention scores** | **`4·s²·d`** | **Flash Attention** |
| O projection | `2·s·d²` | oneDNN GEMM |
| Gate projection | `2·s·d·d_ff` | oneDNN GEMM |
| Up projection | `2·s·d·d_ff` | oneDNN GEMM |
| Down projection | `2·s·d_ff·d` | oneDNN GEMM |

Attention projections scale as O(s·d²) — linear in sequence length.
Attention scores scale as O(s²·d) — **quadratic** in sequence length.
FFN scales as O(s·d·d_ff) — linear in sequence length.

### Attention Score % of Total Layer FLOPs by Model Size and Sequence Length

| Seq Length | 7B | 32B | 72B |
|---|---|---|---|
| 512 | 1.5% | 1.0% | 0.9% |
| **1024** | **3.1%** | **2.1%** | **1.9%** |
| **2048** | **6.0%** | **4.2%** | **3.7%** |
| 4096 | 11.2% | 8.1% | 7.1% |
| 8192 | 20.1% | 15.0% | 13.3% |
| 16384 | 33.8% | 26.6% | 24.0% |
| 32768 | 50.4% | 42.1% | 38.8% |

**Key insight**: Larger models have a *lower* attention score fraction because
their FFN and projection GEMMs grow faster (with d² and d·d_ff) than attention
scores (which grow with s²·d). The 72B model's d_ff/d ratio is 3.61× (vs 5.29×
for 7B), but its absolute d is 2.3× larger, so projections still dominate.

### Crossover: When Scores Exceed Projections (Within Attention Block)

Scores dominate projections when `s > d + d_kv`:

| Model | d | d_kv | Crossover seq_len |
|---|---|---|---|
| 7B | 3584 | 512 | **4,096** |
| 32B | 5120 | 1024 | **6,144** |
| 72B | 8192 | 1024 | **9,216** |

For 72B at s=1024: only **10% of the attention block** is FA-optimizable. The
other 90% is Q/K/V/O projection GEMMs.

### Training Multiplier

Training (forward + backward) multiplies every component by ~3× (1× forward +
2× backward). The **ratios remain the same** — FA optimizes both forward and
backward attention scores, but projections and FFN also scale identically.

FA's memory benefit (avoiding O(s²) attention matrix storage for backward) is
orthogonal to compute — it reduces peak memory, enabling larger batch sizes.

---

## When Flash Attention Actually Matters

### Impact Analysis for Current Workloads

**Qwen2.5-72B GRPO on 3 nodes (84.6 s/step, ~10s generation):**
- Sequence lengths: max_seq_len=512, max_generated_tokens=128 → effective s ≈ 640
- At s=640, attention scores ≈ **1.2% of total FLOPs**
- Even if FA made attention scores free: max speedup = **1.2%** (≈1s on 84.6s step)
- The 74s of non-generation training time is dominated by FSDP communication
  (AllGather + ReduceScatter of 72B params over oneCCL/Slingshot) and GEMM compute

**Qwen3-32B GRPO on 1-2 nodes (18-19 s/step):**
- At s ≈ 512, attention scores ≈ **1.0% of total FLOPs**
- Maximum theoretical FA2 benefit: <0.2s per step

### When FA2 Would Provide Meaningful Gains

FA2 becomes impactful when:

1. **Sequence length > 4K-8K tokens** — attention scores reach 7-20% of total FLOPs,
   and O(s²) memory pressure becomes the practical bottleneck (OOM without FA)
2. **Memory-bound, not compute-bound** — FA's primary benefit is reduced HBM traffic;
   on PVC with 3.2 TB/s HBM, math-only SDPA is less bottlenecked than on A100
3. **Attention is the bottleneck** — when communication overhead is small (e.g.,
   single-node or highly overlapped) and generation is pre-computed (e.g., via vLLM)
4. **Sequence packing** — packing multiple variable-length responses into fixed bins
   requires FlashAttention + BlockMask for correct masking; math-only SDPA with
   packed sequences would attend across sample boundaries

### Does Model Size Increase FA Impact?

**No — larger models reduce FA's relative impact.** The attention score fraction
*decreases* with model size (1.9% for 72B vs 3.1% for 7B at s=1024) because
projections and FFN grow as O(d²) and O(d·d_ff) while scores grow only as O(d).

However, larger models on more nodes shift the bottleneck profile:
- **Communication fraction increases** with node count (more AllGather/ReduceScatter)
- **GEMM efficiency may decrease** if per-tile batch size shrinks with more shards
- **Neither of these is helped by FA** — the bottleneck moves further from attention

The only scenario where 72B benefits more from FA than 32B is if you increase
sequence lengths proportionally to model size (e.g., 72B with 8K context), which
pushes attention scores to ~13% of FLOPs.

---

## Other Kernel Optimizations

### oneDNN GEMM (Current Bottleneck)

~96-98% of training FLOPs are linear-layer GEMMs dispatched through oneDNN.
Optimizations in this area have the highest leverage:

| Optimization | Status | Notes |
|---|---|---|
| BF16 XMX utilization | Baseline | oneDNN auto-selects DPAS-based kernels |
| GEMM shape tuning | Not explored | Batch size affects DPAS tile utilization |
| Persistent GEMM kernels | Not available | oneDNN doesn't expose this |
| INT8 quantization (QLoRA) | Not explored | Would halve GEMM memory traffic |

### torch.compile on XPU

| Variant | Result | Notes |
|---|---|---|
| Static shapes (single-node) | Unstable | Recompiles on shape changes (1.5s cached vs 9.7s miss) |
| Dynamic shapes | Worse | ~40s per unique kernel (Triton-to-SYCL compilation) |
| Multi-node | Disabled | Deadlock risk with oneCCL |

**Root cause**: XPU inductor backend compiles Triton→SYCL→Level Zero, fundamentally
slower than CUDA's Triton→PTX path. Variable-length RL sequences trigger constant
recompilation.

### AllGather-Compute Overlap (FSDP2)

| Scope | Result | Notes |
|---|---|---|
| Single-node (10 tiles) | No benefit | AllGather already fast via shared memory |
| Multi-node | Untested | May help across Slingshot fabric |

Enabled via `reshard_after_forward=None` in FSDP2 (PyTorch PR #155319).

### FlexAttention

| Feature | Status | Notes |
|---|---|---|
| XPU support | **Broken** | `_SUPPORTS_FLEX_ATTENTION` checks `torch.cuda.is_available()` — returns False on XPU |
| Compiled via Triton | Possible | Intel Triton benchmarks show FlexAttention running on XPU |
| Use case | Sequence packing | Needed for correct packed attention masking (BlockMask) |

Fix needed: patch `torchtune/utils/_import_guard.py` to check XPU capability.

### Activation Checkpointing

Only practical memory optimization that works on XPU. Trades compute for memory
by recomputing activations during backward pass. Currently enabled for 72B
(`enable_activation_checkpointing: True`).

---

## Test Log

### 2026-04-02: FA2 vs Math-Only SDPA (32B, 10 tiles, G=4)

- **Config**: `force_math_sdpa=False`, fixed CCL (WORKER_COUNT=1, MPI transport),
  no `empty_cache()`
- **Result**: 18.3s vs 18.2s — no improvement, no crashes
- **Conclusion**: UR leak hypothesis disproven; FA2 provides no throughput gain at
  s ≈ 512. Attention scores are ~1% of total FLOPs at this sequence length.

### 2026-04-02: forward_batch_size=4 (72B, 24 tiles, 3 nodes, CPU offload)

- **Config**: `qwen72B_grpo_fbs4.yaml` — fbs=4, AdamW, CPU offload, G=4
- **Hypothesis**: fbs=4 should improve GEMM utilization (matrix-matrix vs matrix-vector),
  yielding ~25-27s savings based on 32B fbs=1→fbs=4 measurements (9-13× forward penalty).

**Result: WORSE with CPU offload — 117-122s vs 84.6s baseline.**

| Step | Total | Gen | GRPO | Opt | vllm | policy_fwd | ref_fwd | backward |
|---|---|---|---|---|---|---|---|---|
| 0 (warmup) | 73.3s | 27.0s | 34.8s | 9.0s | 11.5s | 8.1s | 7.0s | 28.3s |
| 1 | 117.2s | 52.7s | 55.9s | 6.6s | 14.7s | 22.8s | 18.0s | 39.9s |
| 2 | 121.7s | 47.0s | 66.6s | 6.1s | 14.1s | 17.6s | 17.6s | 50.1s |

**Analysis**: fbs=4 + CPU offload creates a negative interaction:
1. Policy/ref forward during logprob computation jumped from ~8s to 17-23s after
   step 0 — CPU→GPU parameter transfers under increased activation memory pressure.
2. Backward time **escalates each step** (28→40→50s) — suggests growing memory
   fragmentation from larger activation footprint + CPU offload thrashing.
3. The GEMM utilization gain from batching 4 sequences is completely negated by
   CPU offload overhead. CPU offload's per-layer parameter transfer is the bottleneck,
   not GEMM shape.

**Conclusion**: fbs=4 requires eliminating CPU offload first. With on-device weights
(no offload), fbs=4 should show the expected GEMM improvement. The optimization
path must be: **remove CPU offload → then increase fbs**.

### 2026-04-02: Adafactor optimizer (72B, 24 tiles, 3 nodes, CPU offload)

- **Config**: `qwen72B_grpo_adafactor.yaml` — Adafactor, fbs=4, CPU offload, G=4
- **Hypothesis**: Adafactor's factored second moment halves optimizer memory, enabling
  no-offload on 36 tiles.

**Result: CRASH — DTensor in-place op incompatibility.**

```
RuntimeError: aten.pow_.Scalar: in-place operations that require placement changes
are not supported. The operation would change placement from
(_NormPartial(reduce_op=sum, norm_type=2),) to (Replicate(),)
```

Adafactor's internal C++ kernels use `pow_` (in-place) on DTensors that FSDP2
shards. The DTensor system can't handle in-place ops that change tensor placement
(from partial to replicate). This is a known DTensor/FSDP2 limitation.

**Workarounds to investigate**:
- [ ] `foreach=False` — force single-tensor (non-fused) path, may avoid the op
- [ ] Custom Adafactor that avoids in-place ops on sharded state
- [ ] Use `torch.optim.Adam` with `amsgrad=False` and BF16 state (still 2 moments)
- [ ] 8-bit Adam via custom implementation (no bitsandbytes XPU backend)

### 2026-04-02: 32B dedicated vLLM (2 nodes: 1 vLLM + 1 training)

- **Config**: `qwen32B_grpo_dedicated_vllm_xpu.yaml` — 12 training tiles, fbs=4, G=4, TP=4 DP=3 vLLM
- **Baseline**: 18.2s/step (10+2 colocated single-node)

| Step | Total | Gen | GRPO | Opt | vllm | policy_fwd | ref_fwd |
|---|---|---|---|---|---|---|---|
| 0 (warmup) | 22.2s | 12.0s | 7.8s | 1.5s | 7.1s | 3.0s | 1.5s |
| 1 | 16.5s | 10.4s | 5.8s | 0.2s | 6.9s | 2.2s | 1.4s |
| 2 | 16.9s | 10.5s | 6.2s | 0.2s | 6.8s | 2.2s | 1.5s |
| 3 | 16.4s | 10.3s | 5.8s | 0.2s | 6.8s | 2.1s | 1.4s |
| 4 | 16.6s | 10.4s | 6.0s | 0.2s | 6.8s | 2.1s | 1.4s |

**Result: 16.4-16.9s/step steady state (~10% faster than colocated baseline).**
- Training (GRPO) improved: 5.8-6.2s vs ~6.5s (12 tiles vs 10)
- Generation slightly slower: 10.3-10.5s vs ~10s (cross-node HTTP adds ~0.5s latency)
- Optimizer on-device: 0.2s (was already on-device in baseline)
- No memory issues, stable across all 5 steps

### 2026-04-02: 72B no-offload (4 nodes: 1 vLLM + 3 training, fbs=4)

- **Config**: `qwen72B_grpo_no_offload.yaml` — 36 training tiles, fbs=4, no CPU offload
- **Baseline**: 84.6s/step (24 tiles, CPU offload, fbs=1)

| Step | Total | Gen | GRPO | Opt | Notes |
|---|---|---|---|---|---|
| 0 (warmup) | 59.3s | 28.8s | 27.7s | 2.1s | Completed successfully |
| 1 | CRASH | — | — | — | GPU segfault (OOM) on backward |

**Result: OOM on step 1.** Step 0 completed at 59.3s — a **30% improvement** over
baseline (84.6s) — confirming that removing CPU offload + fbs=4 is the right path.
The crash occurred during backward pass on step 1 (longer sequence = more activation
memory). Peak memory after model init was 9.90 GiB; the remaining ~38 GiB was
insufficient for optimizer states (16 GiB) + activations (3.8 GiB) + gradients
(4 GiB) + AllGather buffer (1.8 GiB) + backward overhead.

**Next steps to fix OOM**:
- [x] Reduce fbs to 2 — tested, still OOM (see below)
- [x] Between-step `empty_cache()` — implemented, testing
- [ ] Use gradient checkpointing more aggressively (per-layer vs per-block)
- [ ] INT8 reference model (saves ~2 GiB/tile)
- [ ] Increase `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` (cache warnings seen)

### 2026-04-02: 72B no-offload fbs=2 (4 nodes, without empty_cache fix)

- **Config**: `qwen72B_grpo_no_offload_fbs2.yaml` — 36 tiles, fbs=2, no CPU offload
- **Hypothesis**: fbs=2 halves activation memory vs fbs=4, should avoid OOM

| Step | Total | Gen | GRPO | Opt | Notes |
|---|---|---|---|---|---|
| 0 | 72.8s | 43.1s | 27.1s | 2.0s | Completed (vllm=11.8s, policy_fwd=16.5s, ref_fwd=14.4s, grpo_fwd=8.3s, bwd=18.8s) |
| 1 | CRASH | — | — | — | GPU segfault during policy forward (step 1) |

**Result: Same OOM pattern as fbs=4 — step 0 works, step 1 crashes.** Step 0 was
72.8s (14% faster than 84.6s baseline), confirming no-offload works but slower than
fbs=4's 59.3s due to less GEMM batching.

**Root cause**: NOT memory fragmentation — it's **insufficient physical memory**.
Memory diagnostics with between-step `empty_cache()` revealed:

| Metric | Value |
|---|---|
| Step 0 PRE-STEP allocated | 7.60 GiB (models only, no optimizer) |
| Step 0 peak reserved | **50.5 GiB** (exceeds 48 GiB physical!) |
| Post-step 0 allocated | 19.78 GiB (models + optimizer states) |
| empty_cache freed | 28.13 GiB cached blocks |
| Step 1 PRE-STEP allocated | 19.78 GiB |
| Step 1 result | OOM during policy forward (even with fbs=1!) |

Step 0 works because optimizer states (12 GiB) aren't allocated until the first
`optimizer.step()`. Step 0's forward/backward peaks at ~50.5 GiB with only 7.6 GiB
baseline. Step 1 starts with 19.78 GiB baseline (models + optimizer states) —
the same peak forward/backward would need 50.5 - 7.6 + 19.78 = **62.7 GiB**,
far exceeding 48 GiB.

**Conclusion**: 36 tiles (3 training nodes) is fundamentally insufficient for 72B
without CPU offload. The optimizer states (12 GiB/tile for AdamW FP32) consume too
much of the 48 GiB tile budget. Solutions:
1. **More tiles**: 48 tiles (4 training nodes = 5 total) reduces per-tile to ~25 GiB
2. **Memory-efficient optimizer**: SGD (no state), Adafactor (half the state)
3. **INT8 reference model**: Saves ~2 GiB/tile

### 2026-04-02: 72B no-offload fbs=1 (4 nodes, with empty_cache + memory diag)

- **Config**: `qwen72B_grpo_no_offload_fbs1.yaml` — 36 tiles, fbs=1, no CPU offload
- **Result: OOM on step 1, same as fbs=2.** Confirms the issue is baseline memory
  (optimizer states), not activation memory.

| Step | Total | Gen | GRPO | Opt | Notes |
|---|---|---|---|---|---|
| 0 | 100.4s | 69.6s | 27.4s | 2.0s | policy_fwd=31.0s, ref_fwd=27.5s (slower than fbs=2) |
| 1 | CRASH | vllm=10.3s | — | — | OOM during policy forward |

### 2026-04-02: 72B no-offload 5 nodes (48 tiles)

- **Config**: `qwen72B_grpo_no_offload_5node.yaml` — 48 tiles, fbs=2, no CPU offload
- **Result: OOM on step 1, same as 36 tiles.**

| Step | Total | Gen | GRPO | Opt | Notes |
|---|---|---|---|---|---|
| 0 | 53.2s | 28.3s | 22.2s | 2.0s | Peak reserved 47.2 GiB — barely under 48 GiB |
| 1 | CRASH | vllm=11.4s | — | — | OOM during policy forward (same as 36 tiles) |

**Root cause confirmed**: The peak memory overhead during forward/backward is
**~41 GiB regardless of tile count** because FSDP AllGather materializes full
unsharded layers (1.63 GiB per layer × multiple layers in flight). This overhead
is constant — adding tiles only reduces per-tile shard size, not AllGather buffer.

Memory equation: `baseline_after_optimizer + 39 GiB ≤ 48 GiB`
→ baseline ≤ 9 GiB → need ~90 tiles (8 training nodes, 9 total)

### 2026-04-02: Within-forward empty_cache causes UR handle crash

Attempted `torch.xpu.empty_cache()` between forward pass chunks (between model
calls) to free cached allocator blocks. Result: GPU segfault crash.

**Confirmed**: `empty_cache()` is NOT safe between model forward calls on XPU
because FSDP2 keeps AllGather prefetch buffers active even between model calls.
The only safe point for `empty_cache()` on XPU with FSDP is **between training
steps** when FSDP is fully idle (all parameters sharded, no prefetch).

### Conclusion: 72B No-Offload Requires 8+ Training Nodes

72B without CPU offload on XPU (48 GiB tiles) is not practical with ≤4 training
nodes. The FSDP AllGather + forward/backward overhead of ~41 GiB leaves only
~7 GiB for model shards + optimizer states.

**Practical paths forward for 72B optimization:**
1. **Keep CPU offload, optimize other bottlenecks** — generation time (10s→faster vLLM),
   optimizer step (6.5s→faster CPU-GPU transfer), backward (18s→fewer AllGather ops)
2. **Optimizer-only CPU offload** — keep params on-device for fast forward/backward,
   offload only AdamW states. Requires custom implementation (not built into FSDP2).
3. **Hybrid: fbs=1 + CPU offload + more tiles** — current baseline with more nodes
   to reduce per-tile overhead
4. **8-bit optimizer states** — halve optimizer memory (requires custom or bitsandbytes)
5. **SGD optimizer** — no persistent state, but much worse convergence

### Pending Tests

- [ ] Adafactor with `foreach=False` to avoid DTensor in-place crash
- [ ] FA2 at longer sequence lengths (4K+) — expected to show measurable gain
- [ ] Intel Triton FA2 kernel vs PyTorch SYCL-TLA FA2 — may have better tuning
- [ ] FlexAttention on XPU via Triton backend — requires _import_guard.py fix
- [ ] AllGather-compute overlap on multi-node (2+ nodes over Slingshot)
- [ ] GEMM shape profiling — identify per-tile batch sizes that maximize DPAS utilization
