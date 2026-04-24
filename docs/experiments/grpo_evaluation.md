# GRPO Evaluation Loop

Periodic evaluation on held-out data during GRPO training, measuring whether the model is actually learning to solve problems (not just throughput benchmarks).

## Overview

The eval loop is built into `recipes/dev/grpo_full_finetune_distributed_xpu.py`. Every N training steps, it generates responses to held-out GSM8K test problems via the same vLLM server used for training, computes rewards, and logs `eval/` metrics alongside training metrics.

All ranks participate in eval (required for the vLLM broadcast pattern), but only rank 0 logs. Works on both single-node and multi-node (OFI transport required for multi-node Aurora).

## Config Parameters

```yaml
# Add to any GRPO XPU config
eval_dataset:
  _component_: torchtune.dev.grpo.gsm8k.gsm8k_dataset
  split: test                  # GSM8K test split (1319 examples)
eval_every_n_steps: 20         # run eval every N training steps
eval_max_examples: 10          # subsample from test set (controls eval time)
eval_grpo_samples: 4           # completions per eval problem (can differ from training G)
```

- `eval_dataset`: Any dataset compatible with `gsm8k_dataset` signature. Uses HF `split` parameter. If omitted, eval is disabled.
- `eval_every_n_steps`: Must be > 0 to enable eval. Eval runs at steps N, 2N, 3N, ... and at end of training.
- `eval_max_examples`: First N examples from the eval dataset. Each takes ~21s at max_gen=256 (vLLM generation dominates).
- `eval_grpo_samples`: Number of completions generated per problem. Lower than training G saves time since eval doesn't need diverse samples for advantage estimation.

## Eval Timing Budget

Each eval example requires `eval_grpo_samples` vLLM generation calls (batched). Cost depends on model size, vLLM TP, and max_gen_tokens.

**32B model, TP=2 (single-node or 2-node DP vLLM):**

| eval_grpo_samples | max_gen=256 | max_gen=512 | 10 examples @ 512 |
|---|---|---|---|
| 2 | ~20s/example | ~40s/example | ~6 min |
| 4 | ~40s/example | ~80s/example | ~13 min |

**3B model, TP=1 (single-node, 2 DP vLLM replicas):**

| eval_grpo_samples | max_gen=256 | max_gen=512 | 10 examples @ 512 |
|---|---|---|---|
| 4 | ~5s/example | ~8s/example | ~1.3 min |

**Recommendation**: Use 10 examples with `eval_grpo_samples: 2` for iterative runs (6 min eval overhead for 32B). Use 50+ examples with `eval_grpo_samples: 4` for publication-quality runs. On debug queue (1-hour walltime), budget ~6 min per eval cycle when planning step count.

## Logged Metrics

Training metrics (logged every step):
```
METRICS step=N  loss=...  policy_loss=...  kl_loss=...  rewards=...  successes=...
                grad_norm=...  clipfrac=...  ratios=...  approx_kl=...  resp_len=...
```

Eval metrics (logged every `eval_every_n_steps`):
```
EVAL step=N  rewards=...  successes=...  resp_len=...  time=...s  (K examples)
```

| Metric | Meaning |
|--------|---------|
| `successes` | Fraction of completions where `math_verify` confirms correct answer |
| `rewards` | Sum of all reward functions (math correctness + formatting) |
| `resp_len` | Average response length in tokens after stop-token truncation |
| `time` | Wall-clock seconds for the full eval pass |

Both training and eval metrics are written to DiskLogger at `{output_dir}/logs/log_*.txt`.

## How It Works

1. At the configured step interval, `run_eval()` is called from the training loop (after `cleanup_after_step`, before the next batch).
2. Each eval example is processed sequentially (batch_size=1):
   - Collate and tokenize the prompt
   - Expand by `eval_grpo_samples` completions
   - Generate via `_generate_with_vllm()` (rank 0 calls HTTP server, broadcasts to all ranks)
   - Truncate at stop tokens
   - Compute rewards via `batched_rewards()` (same reward functions as training)
   - Accumulate reward and success statistics
3. Rank 0 logs aggregated metrics with `eval/` prefix.
4. Training resumes from the next batch.

## Key Design Decisions

**Why not a separate eval process?** Eval reuses the running vLLM server and FSDP model. A separate process would need its own checkpoint loading, separate vLLM, and coordination. The in-loop approach is simpler and guarantees eval uses the exact current model weights.

**Why no policy/ref forward pass in eval?** Training trajectory generation computes policy and reference log-probabilities for the GRPO loss. Eval only needs rewards (did the model get the right answer?), so we skip the expensive FSDP forward passes. This makes eval ~3x faster per example than a full training step.

**Why sequential examples (not batched)?** Memory safety. Each eval example generates `eval_grpo_samples` sequences through vLLM. Batching multiple eval problems would multiply the vLLM batch size and could OOM. Sequential processing reuses the same memory.

## Validation Run Results

### Run 1: Qwen3-32B (2026-04-05, 24 steps)

First validation run, identified issues.

**Eval at step 20:**
- `eval/successes=0.050` (5% — baseline for 32B with max_gen=256)
- 50 examples, 1076s (18 min)

**Issues found and fixed:**
1. frameworks/2025.3.1 caused `cxil_map: write error` crash — fixed to 2025.2.0
2. Eval too slow at 50 examples — reduced to 10
3. lr=1e-6 too low to see learning in <100 steps — increased to 5e-6
4. `eval/resp_len` reported 1.8 instead of ~255 — truncation mask bug fixed (inverted padding mask)

### Run 2: Qwen2.5-3B with vLLM DP=2 (2026-04-05, 50 steps)

Fast iteration config: G=16, fbs=16, max_gen=512, lr=1e-5, 2 vLLM replicas (DP=2, TP=1).
10 training tiles, 2 vLLM tiles. Memory: 18.8 GiB active, 53 GiB reserved.

**Eval progression — learning signal confirmed:**

| Step | eval/successes | eval/rewards | eval/resp_len | eval/time |
|------|---------------|-------------|---------------|-----------|
| 20   | 0.475         | 51.4        | 133.3         | 49s       |
| 40   | 0.525         | 56.4        | 150.0         | 63s       |

+5 percentage points on held-out test over 20 training steps. Responses getting longer as model learns chain-of-thought.

**Training metrics (steps 1-50):**
- `successes` fluctuated 0.31-0.88 per batch (noisy with batch_size=1)
- `grad_norm` spiked to 444 at step 10 (warmup ending, lr reaching peak), settled to 7-66
- `kl_loss` rose from 0.0006 → 0.020 (expected: model diverging from reference)
- `resp_len` ranged 74-280 (not hitting 512 truncation — good)

**Known issue:** `save_every_n_steps: 50` crashes during FSDP2 checkpoint gathering. Disabled for now.

### Run 3: Qwen3-32B, 2-node multi-node (2026-04-06, 35 steps)

First successful multi-node GRPO run. 2 nodes × 10 training tiles + 2 × TP=2 vLLM.
Flat FSDP (dp_replicate=1) across 20 ranks, OFI transport (MPI transport deadlocks on
multi-node Aurora — see below). Rank 0 dispatches to both nodes' vLLM servers in parallel.

**Critical multi-node fixes required:**
1. `CCL_ATL_TRANSPORT=ofi` — MPI transport deadlocks during XCCL communicator creation
   (mpi4py pre-init + CCL MPI transport = conflicting KVS paths, only one rank proceeds)
2. `no_proxy="*"` — Aurora's Squid proxy blocks inter-node HTTP to vLLM on port 8001
3. CPU-local-sharding for checkpoint loading — `distribute_tensor` scatter collectives
   also deadlock on XCCL; compute shard offsets on CPU and construct DTensor without collectives
4. Use `localhost` for local vLLM URL, HSN FQDN for remote nodes
5. `sampler_replicas=1, sampler_rank=0` in vLLM server mode — all ranks must see the
   same batch for broadcast shape matching

**Config:** G=8, fbs=8, max_gen=512, lr=5e-6, kl_coeff=0.05, eval_every_n_steps=25,
eval_max_examples=10, eval_grpo_samples=2.

**Eval progression — no improvement in 35 steps:**

| Step | eval/successes | eval/rewards | eval/resp_len | eval/time |
|------|---------------|-------------|---------------|-----------|
| 25   | 0.550         | 58.1        | 426.0         | 363s      |
| 35   | (killed by walltime during eval generation) | | | |

55% accuracy at step 25 matches baseline (step 5 also showed 55% during validation runs).
35 steps with batch_size=1 means the model only saw ~35 unique problems out of 7,473 in
GSM8K train — insufficient for measurable eval improvement on a 32B model.

**Training metrics (35 steps):**
- KL divergence rising steadily: 0.0005 (step 1) → 0.025-0.060 (steps 30-35), confirming
  the model is learning/deviating from reference. Spike to 0.193 at step 22.
- Training successes: ~1.5/8 per step (~19%), noisy, no clear upward trend
- `grad_norm`: mostly 0.9-1.4, spike to 4.9 at step 22 (recovered)
- `policy_loss` near zero throughout (expected early in training)
- Average step time: ~76s (vLLM 51s + GRPO 24s + optimizer 0.1s)
- Memory: 15.19 GiB allocated, 54.50 GiB reserved per rank

**Step timing breakdown (2-node, OFI transport):**

| Component | Time | Notes |
|-----------|------|-------|
| vLLM generation | 30-55s | 2 DP servers, varies with response length |
| Broadcast | <0.1s | World PG, OFI transport |
| Policy forward | 5.8-6.2s | 20-way FSDP |
| Ref forward | 5.0-5.4s | 20-way FSDP |
| GRPO backward | ~18s | Includes ReduceScatter |
| Optimizer | 0.1s | AdamW |
| **Total** | **66-81s** | Avg ~76s steady-state |

**Performance comparison:**

| Config | Step time | vLLM tok/s | Notes |
|--------|-----------|-----------|-------|
| Single-node (10+2 tiles) | 22.8s | ~55 | MPI transport, rank-local vLLM |
| 2-node, 1 vLLM (v14) | 72-80s | 49-55 | OFI transport, rank-0 only |
| 2-node, 2 DP vLLM (v17) | 66-81s | 61-85 | OFI transport, round-robin dispatch |

2-node is ~3.3x slower than single-node due to: (1) OFI transport has ~2x lower
intra-node AllGather bandwidth than MPI, (2) only rank 0 calls vLLM (no shard-level
broadcast), (3) inter-node FSDP communication overhead. The benefit of multi-node is
running models that don't fit on one node, not throughput scaling.

## Plotting

```bash
python recipes/dev/plot_learning_curves.py results/qwen32b_learning_run/logs/
# Produces learning_curves.png with train/eval success rate, rewards, loss
```

Compare multiple runs:
```bash
python recipes/dev/plot_learning_curves.py \
  results/qwen32b_learning_run/logs/ \
  results/gemma4_learning_run/logs/ \
  --labels "Qwen3-32B" "Gemma4-31B" \
  -o model_comparison.png
```
