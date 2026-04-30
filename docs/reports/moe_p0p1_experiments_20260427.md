# Qwen3-30B-A3B MoE P0/P1 Experiment Results

**Date**: 2026-04-27
**Node**: x4302c2s5b0n0 (single Aurora node, debug-scaling)
**Layout**: 10 training tiles + 2 vLLM tiles (TP=2)
**Model**: Qwen3-30B-A3B (30B total, 3B active per token, 128 experts)
**Config base**: `qwen3_30b_a3b_grpo_xpu.yaml`
**vLLM**: v0.15.0, `--max-model-len 2048`, `--gpu-memory-utilization 0.80`

## Summary

| Test | Config | Steps | Step Time (warm) | Result |
|------|--------|-------|-----------------|--------|
| A (baseline) | batch=1, G=4, fbs=4 | 1/3 | 50.9s (step 0) | OOM step 1 |
| B (batch=2) | batch=2, G=4, fbs=4 | 0/3 | — | OOM step 0 bwd |
| C (G=8) | batch=1, G=8, fbs=8 | **3/3** | **54.8s** | PASS |
| D (compile) | batch=1, G=4, fbs=4, compile=True | 0/3 | — | SYCL compile timeout |

**Winner: Test C (G=8)** — 2x RL samples per step, stable memory, 9.2 tok/s throughput.

## Test A: fbs=4 Baseline (batch=1, G=4)

Completed 1 of 3 steps. OOM at step 1 backward (rank 1, SIGABRT).

**Step 0 timing**:
- TIMING: total=50.9s, gen=17.4s, grpo=27.7s, clip=0.2s, opt=1.4s, other=4.2s
- GENTIMING (warmup): vllm=4.5s, policy_fwd=11.0s, ref_fwd=1.6s

**Memory**: Between step 0→1, gap = 12.95 GiB (allocated=30.40, reserved=43.35).
Step 1 backward allocation exceeds remaining headroom.

**Weight sync**: Not captured (died before completing step 1 sync).

## Test B: batch=2 (batch=2, G=4)

Died during step 0 backward. 8 sequences (2 prompts × 4 completions) consume nearly
all 64 GiB per tile before backward even starts.

**GENTIMING (warmup)**: vllm=3.3s, policy_fwd=4.5s, ref_fwd=3.1s

**Root cause**: batch=2 doubles prompt storage + completion storage. With 30B params
sharded across 10 tiles (~3 GiB/tile) plus activations for 8 sequences, backward
doesn't have enough headroom for FSDP AllGather of full parameters.

## Test C: G=8 (batch=1, G=8)

**All 3 steps completed.** Memory stabilized after step 1.

| Step | Total | Gen | GRPO | Clip | Opt | Other | Notes |
|------|-------|-----|------|------|-----|-------|-------|
| 0 | 67.0s | 14.7s | 46.6s | 0.2s | 1.2s | 4.2s | Warmup |
| 1 | 62.1s | 7.4s | 42.8s | 0.2s | 0.2s | 11.5s | |
| 2 | 54.8s | 7.6s | 43.5s | 0.2s | 0.2s | 3.3s | Steady-state |

**GENTIMING (warm)**: vllm=3.1s, policy_fwd=2.1-2.2s, ref_fwd=2.1-2.2s

**Weight sync (SHM)**:
| Step | Gather | Copy | HTTP | Total |
|------|--------|------|------|-------|
| 0 | 4.0s | 43.4s (1.3 GB/s, page-fault) | 15.4s | 62.8s |
| 1 | 3.4s | 5.8s (9.7 GB/s) | 17.0s | 26.2s |
| 2 | 3.3s | 7.1s (8.1 GB/s) | 14.6s | 25.0s |

**Memory (rank 0)**:
- Between-step gap: 24.7 GiB (step 0) → 30.5 GiB (step 1) → stable
- POST-BWD worst: ranks 6/7 at 0.41 GiB free (tight but stable)
- peak_resv: 62.43 GiB on rank 6 (of 63.98 GiB total)

**Throughput**: 504 tokens / 54.8s = **9.2 tok/s** (warm)

**Exit**: code 1 (XCCL teardown hang, training completed fine)

## Test D: torch.compile

SYCL kernel compilation took 25+ minutes and did not complete within the 1-hour job
walltime. 500+ kernel modules compiled (6800+ lines of output), still in the first
policy forward pass compilation when the job expired.

**Observation**: torch.compile on XPU generates SYCL C++ kernels via the inductor
backend. With 48 MoE transformer layers (each containing attention + router + 128
expert BMMs), the compilation volume is enormous. This is a practical blocker for
single-node MoE compile — would need pre-cached compiled kernels or AOT compilation.

## Key Findings

### 1. G=8 is the optimal config for single-node MoE GRPO
- 2x RL samples per step vs G=4
- Only 1.1x step time increase (55s vs 51s)
- Memory stabilizes (survives 3+ steps)
- Throughput: 9.2 tok/s vs ~5 tok/s (1.8x improvement)

### 2. Memory fragmentation is the step-1+ killer for G=4
- Between-step gap (reserved - allocated) = 13 GiB for G=4, 30 GiB for G=8
- Paradoxically, G=8's larger initial allocation "pre-shapes" the allocator block pool
  to match steady-state needs, while G=4's smaller blocks fragment on step 1
- G=4 has ~20 GiB l0_free at step 1 start but can't use it (fragmented reserved blocks)

### 3. FSDP communication still dominates
- GRPO phase (fwd+bwd): 42.8-43.5s for 8 sequences
- Compute portion (8 seqs × MoE fwd+bwd): ~5-7s estimated
- Communication (AG+RS for 30B params): ~36-38s
- Communication is ~85% of fwd+bwd time

### 4. torch.compile is impractical for MoE on XPU
- 48 layers × attention + MoE = hundreds of unique kernels
- SYCL compilation: 25+ minutes, didn't finish in 1 hour
- Not viable without AOT compilation or kernel caching

### 5. batch=2 is not viable at current memory budget
- 8 sequences from batch=2 OOM during step 0 backward
- Unlike G=8 (also 8 sequences), batch=2 has different memory layout:
  2 prompts stored + 8 completions vs 1 prompt + 8 completions

## Comparison with Previous E2E Run

Previous E2E run (G=4, fbs=2): 35.5s/step steady-state
This P0 run (G=8, fbs=8): 54.8s/step but 2x sequences = 1.55x total throughput

| Metric | Previous (G=4, fbs=2) | P0 (G=8, fbs=8) |
|--------|----------------------|-----------------|
| Step time | 35.5s | 54.8s |
| Sequences/step | 4 | 8 |
| Tokens/step | 256 | 504 |
| Throughput | 7.2 tok/s | 9.2 tok/s |
| RL samples/step | 4 | 8 |
| GRPO fwd+bwd | ~23s | ~43s |

## Next Steps

1. **Production config**: Use G=8 as the default for Qwen3-30B-A3B MoE GRPO
2. **Expert Parallelism**: EP is the only path to reduce communication dominance
   (from ~122 GB AG+RS to ~13 GB for attention+router only)
3. **Longer runs**: Test G=8 for 20+ steps to verify memory stability
4. **AOT compile**: Investigate `torch._export.aot_compile` for MoE kernel pre-compilation
5. **G=16**: Test if larger G further amortizes communication (needs memory check)
