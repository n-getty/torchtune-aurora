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

## Step-time / throughput (2-node)

Accuracy is one axis; wall-clock per step is the other. Two important caveats up front so
the two tables below aren't mis-read against each other:

- **The torchtune topology study ran at `G=16`; the ezpz runs only exist at `G=4`.** So the
  only *matched-envelope* ezpz-vs-torchtune step-time point is the `G=4` flat-colocate row
  (Table A). The full torchtune topology spread (Table B) is `G=16` and is **not** directly
  comparable to the ezpz number — it shows where torchtune's own best 2N config lands.
- Step time is **not** monotonic in `G` here: flat-colocate is 11.4 s at `G=4` but 7.8 s at
  `G=16`, because the `G=4` sweep used a different backward-chunking config. Do not infer
  ezpz's `G=16` time from its `G=4` time, or cross-read the two tables.

**Table A — matched envelope (`B=1, G=4, max_gen=64`, 24 ranks).** The only apples-to-apples
step-time comparison:

| stack | s/step |
|-------|-------:|
| ezpz/TRL 2N (HF `.generate()`) | ~5.7 |
| torchtune 2N — flat colocate (`dp_replicate=1`) | ~11.4 |

At matched `G=4`, ezpz is ~2× faster per step — but flat colocate (`dp_replicate=1`) is
torchtune's **slowest** 2N topology: it shards one model 24-way *across both nodes*, so
every layer's collective crosses the inter-node link. A matched-`G=4` run of torchtune's
*best* topology hasn't been done; Table B shows that best at `G=16`.

**Table B — torchtune 2N topology spread (`G=16`, internal study, NOT matched to ezpz).**
From `docs/reports/agpt2b_2n_topology_throughput_20260618.md`:

| torchtune 2N topology | dp_rep | train tiles | s/step | best success |
|-----------------------|:------:|:-----------:|-------:|:------------:|
| flat colocate          | 1 | 24 | 7.8 | 0.80 |
| **colocate-HSDP**      | **2** | **24** | **5.5** | **1.00** |
| 11+1 per-node          | 2 | 22 | 5.9 | 0.44* |
| dedicated vLLM node    | 1 | 12 | 7.6 | 0.50* |

\* server-mode topologies (11+1, dedicated) at their stable envelope; lower convergence is
a topology property (fewer training ranks / distinct prompts), not instability — all 0 NaN.

**The headline from Table B:** the dominant 2N lever is the FSDP **replicate dimension**, not
vLLM placement. `dp_replicate=2` (two intra-node 12-way shards, one replica per node) keeps
the heavy shard collective on-node and is both the fastest (**5.5 s/step, −29% vs flat**) and
the most accurate (best_acc 1.00). So torchtune's *best* 2N configuration is ~5.5 s at `G=16`
— in the neighbourhood of ezpz's `G=4` 5.7 s, while carrying the accuracy/stability edge from
the sweep above.

> **What's not yet measured (honest gap):** a matched-`G=4` `dp_replicate=2` torchtune run.
> Until that exists, "torchtune's best 2N ≈ ezpz on step time" is **directional** (different
> `G`), not a clean like-for-like throughput claim. The **accuracy** comparison is fully
> matched (`B=1, G=4`). ezpz could not be run at <24 ranks (single-node crashes early on an
> Intel L0 event-pool exhaustion in HF Trainer's per-step cache cleanup), so all ezpz cells
> are 24-rank.

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
