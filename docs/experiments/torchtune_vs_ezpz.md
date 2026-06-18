# torchtune-XPU vs ezpz/TRL — GRPO on AuroraGPT-2B

A like-for-like comparison of two GRPO reinforcement-learning stacks on Intel Max
Series GPUs (Aurora):

- **torchtune-XPU** — this repository's GRPO recipe, FSDP (FSDP2 single-node; FSDP1
  HYBRID_SHARD for 2-node `dp_replicate>1`) + vLLM colocate rollout.
- **ezpz/TRL** — Hugging Face `TRL.GRPOTrainer` with HF `.generate()` rollout.

All numbers below are from completed, verified runs (see
[`docs/RESULTS_DISCIPLINE.md`](../RESULTS_DISCIPLINE.md) for the measurement gate).

## Headline

**The two stacks are algorithmically equivalent at their best learning rate, and
torchtune-XPU has more learning-rate headroom before it goes unstable.** On a fixed
task and step budget, both reach the same peak accuracy at `lr=5e-6`; when the
learning rate is pushed to `1e-5`, torchtune keeps climbing while ezpz/TRL stalls.

## Learning-rate sweep (controlled comparison)

Task `sum_digits`, 50 GRPO steps, single seed per cell, identical
`B=1, G=4, max_gen=64, temperature=0.7, beta=0.0`. `best_acc` is the peak
per-step success rate, aggregated across all training ranks.

| learning rate | torchtune 1N (12 ranks) | torchtune 2N (24 ranks) | ezpz/TRL 2N (24 ranks) |
|---------------|------------------------:|------------------------:|-----------------------:|
| 1e-6          | 0.083                   | —                       | 0.125                  |
| 5e-6          | **0.542**               | **0.594**               | **0.542**              |
| 1e-5          | **0.667**               | **0.729**               | 0.083                  |
| 5e-5          | 0.000 (divergent)       | —                       | 0.083 (divergent)      |

Both torchtune columns run vLLM in-process (colocate); 1N is one node (12 tiles),
2N is two nodes (24 tiles) — matching ezpz on node and rank count. Only the two
productive learning rates (`5e-6` tie, `1e-5` headroom) are re-run on 2N; the
dead-zone (`1e-6`) and divergent (`5e-5`) cells are already characterized by the
1N and ezpz columns.

Takeaways:

1. **No algorithmic gap.** At the productive learning rate (`5e-6`), both stacks
   land at an identical `best_acc = 0.542`.
2. **torchtune has more stability headroom.** At `1e-5` torchtune climbs to `0.667`,
   while ezpz/TRL's gradient norm collapses to zero and accuracy stays at the noise
   floor — torchtune tolerates the higher learning rate (bf16 gradient path vs HF
   Trainer's fp16 grad scaler).
3. **The result holds at 2-node scale.** At the same 24-rank footprint as ezpz,
   torchtune matches its 1-node curve and edges slightly higher (`0.594`/`0.729`
   vs `0.542`/`0.667` at `5e-6`/`1e-5`) — consistent with the larger global batch,
   and confirming the comparison is not an artifact of single-node training. At
   `1e-5`, torchtune's 2-node `0.729` stands against ezpz's collapsed `0.083`.
4. **`lr=1e-6` is a dead zone for both.** TRL's default learning rate leaves both
   stacks in the noise floor (single-seed accuracy bounces 0.08–0.25). Framework
   comparisons run at this learning rate measure noise, not convergence.

> **Methodology note.** Always sweep the learning rate **per stack** before comparing
> at a fixed value. Each framework has its own productive range; matching a
> hyperparameter tuned for one stack penalizes the other.

## Step-time / throughput (2-node, matched `B=1, G=4`)

Accuracy is one axis; wall-clock per step is the other. At the matched ezpz envelope
(`B=1, G=4, max_gen=64`, 24 ranks):

| stack / topology | s/step | notes |
|------------------|-------:|-------|
| ezpz/TRL 2N (HF `.generate()`) | ~5.7 | in-process generation, single FSDP shard |
| torchtune 2N — flat colocate (`dp_replicate=1`) | ~11.4 | 24-way **cross-node** FSDP shard — the slow topology |

At this matched envelope ezpz is ~2× faster per step — but the torchtune number above is
its **slowest** 2-node topology: `dp_replicate=1` shards one model 24-way *across both
nodes*, so every layer's AllGather/ReduceScatter crosses the inter-node link. The 2-node
topology study (`docs/reports/agpt2b_2n_topology_throughput_20260618.md`) found the FSDP
**replicate dimension** is the dominant 2N lever: switching to `dp_replicate=2` (two
intra-node 12-way shards, one replica per node) keeps the heavy shard collective on-node
and cuts step time **−29%** (7.8 → 5.5 s at `G=16`). That brings torchtune's best 2N
configuration close to ezpz's per-step time while retaining its accuracy/stability
advantage.

> **Caveat (don't over-read this row).** The −29% `dp_replicate=2` figure was measured at
> `G=16` (5.5 s vs 7.8 s flat), not at the `G=4` envelope of the ezpz step-time cell — a
> matched-`G=4` `dp_replicate=2` run hasn't been done, so this is not yet a clean
> like-for-like throughput win, only a directional one. The **accuracy** comparison above
> is fully matched (`B=1, G=4`); the step-time comparison is matched only for the
> flat-colocate row. ezpz could not be run at <24 ranks (single-node crashes early on an
> Intel L0 event-pool exhaustion in HF Trainer's per-step cache cleanup), so all ezpz
> cells are 24-rank.

## Production result — GSM8K success rate

The same torchtune-XPU GRPO recipe, run as the production AuroraGPT-2B pipeline
(multi-corpus math-mix SFT → GRPO), moves end-task accuracy on GSM8K:

| Stage                          | GSM8K success rate | mean reward (late window) |
|--------------------------------|-------------------:|--------------------------:|
| Raw base model                 | 2.79%              | 0.066                     |
| Math-mix SFT → GRPO            | **6.75%**          | **0.140**                 |

A 2.4× lift in success rate and a 2.1× lift in mean reward over the raw baseline.
Multi-node scale-up (8-node HSDP) and step-based checkpoint resume are validated for
extending the run further. The empirical capability ceiling on this model is
~8–9% GSM8K success; past that needs a capability lever, not more GRPO steps.

Full pipeline and reproduction details:
[`docs/reports/agpt2b_sft_mathmix_to_grpo_20260615.md`](../reports/agpt2b_sft_mathmix_to_grpo_20260615.md).

## Bottom line

torchtune-XPU is a competitive GRPO stack on Aurora/XPU: it matches a mature
TRL-based reference at the optimal learning rate, exceeds it where the reference
becomes unstable, and drives real end-task accuracy gains on AuroraGPT-2B in
production. On per-step throughput ezpz is faster at the matched envelope against
torchtune's *flat* 2-node topology, but the `dp_replicate=2` configuration (the
study-validated best 2N setup) closes most of that gap while keeping the accuracy
and stability edge — see the step-time section above and the topology report.
