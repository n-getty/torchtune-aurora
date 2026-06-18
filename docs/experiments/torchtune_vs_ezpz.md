# torchtune-XPU vs ezpz/TRL — GRPO on AuroraGPT-2B

A like-for-like comparison of two GRPO reinforcement-learning stacks on Intel Max
Series GPUs (Aurora):

- **torchtune-XPU** — this repository's GRPO recipe, FSDP2 + vLLM colocate rollout.
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
production.
