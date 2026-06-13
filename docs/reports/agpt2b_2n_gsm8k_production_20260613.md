# AuroraGPT-2B 2-node GSM8K — production exemplar

**Date**: 2026-06-13
**Runs validated**:
- Engine stability (pre-Stage-1 config): job `8540864` — `experiments/auroragpt_2b_bakeoff/logs/gsm8k_2n_server_stable_20260613_042554/` — **150/150 clean, exit=0, mem flat 24/47 GiB**. This was the previous `_stable` variant: same envelope, but without the Stage 1 vLLM `stop_strings` + EOS-injection fix. Proves the chunked-backward path clears the documented sig#2 L0 wall.
- Stage 1 learning-signal fix (post-fix config, 50-step bisect): held-job `8541808` `stage1c_eosinject_20260613_172749` — **50/50 clean, exit=0, `num_stop_tokens` 0→14.7/16, `response_lengths` 511→111**.
- **Full 150-step production run on Stage-1 fix**: job `8541918` `gsm8k_2n_production_20260613_184607` — **150/150 clean, exit=0**, total wall ~33 min training (~13s/step), `num_stop_tokens` > 0 on **every step**, mean 14.44/16, `response_lengths` mean 121, mem flat 24/58.8 GiB, no NaN/banned:1/UR_RESULT_ERROR.

**Recipe**: `recipes/dev/grpo_full_finetune_distributed_xpu.py`
**Config**:   `recipes/configs/dev/production/auroragpt_2b_grpo_2n_gsm8k_xpu.yaml`
**Launcher**: `experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh`
**Smoke**:    `experiments/auroragpt_2b_bakeoff/smoke_2n_gsm8k.sh` (5-step regression, ~10 min)

> **Reproducibility note**: `experiments/` is gitignored (launcher history
> intentionally not tracked). Reproduce via: 2 nodes on `debug` queue, the
> config + launcher above, model checkpoint at
> `/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650`.

## Summary

This is the first AGPT-2B GRPO production reference on Aurora. Three runs
establish the envelope: a 150-step pre-Stage-1 run proves the engine survives
the documented sig#2 L0 wall (chunked backward; mem flat, exit=0); a 50-step
post-Stage-1 bisect proves the new vLLM `stop_strings` + EOS-injection fix
unblocks the learning signal; and a **150-step production run on the renamed
launcher with the Stage 1 fix** (job `8541918`) shows the same fix holds
across the full envelope — 150/150 clean, exit=0, `num_stop_tokens > 0` on
every step, `response_lengths` mean 121 (vs 511 pre-fix), memory flat.

The Stage 1 vLLM stop-string plumbing fix (`</answer>` / `User:` +
EOS-injection at completion boundary) cut average response lengths from
511 → 111 tokens and finally surfaced `num_stop_tokens > 0` per step —
**without these, every completion ran to `max_tokens` and the model had no
gradient signal worth propagating**.

Reward is non-degenerate but does not yet show a strong monotonic upward trend
in 50 steps. The model is sometimes successful (per-step success rates 0.0–0.19,
peak reward bucket 0.475) and sometimes degenerates into tag-spam loops or
template echoes ("answer here"). This is now a tuning problem, not an engine
problem.

## Production envelope

| Knob                          | Value                                                 |
|-------------------------------|-------------------------------------------------------|
| Topology                      | 2 nodes: 11 train ranks + 12 vLLM HTTP DP=12          |
| Sharding                      | FSDP1 ZeRO-2 (`use_fsdp1_zero2: true`)                |
| `grpo_samples` (G)            | 16                                                    |
| `forward_batch_size`          | 8                                                     |
| `ref_forward_batch_size`      | 8                                                     |
| `max_seq_len`                 | 1024                                                  |
| `max_generated_tokens`        | 512                                                   |
| Optimizer / LR                | AdamW lr=5e-6, warmup=10, cosine, wd=0.01             |
| `clip_grad_norm`              | 1.0                                                   |
| Loss / KL                     | `GRPOSimpleLoss` epsilon=0.2, `kl_coeff=0.02`         |
| Activation checkpointing      | `enable_activation_checkpointing: true`               |
| Reference offload             | `ref_cpu_offload: true`                               |
| vLLM weight sync              | `xccl` intra-node + `gloo` cross-node (TCP over hsn0) |
| `vllm_weight_sync_interval`   | 1 (every step)                                        |
| Attention                     | `TORCHTUNE_USE_IPEX_VARLEN=1` (no-grad path only)     |
| Backward path                 | `TORCHTUNE_USE_CHUNKED_LOSS=0` (chunked train backward)|
| `varlen_nograd_bypass`        | `0` (UNSAFE on un-converged ref — see memory)         |
| `stop_strings`                | `["</answer>", "User:", "\nUser:"]`                  |
| Pinned CPU staging            | `TORCHTUNE_PINNED_CPU_BUF=1`                          |

## Stability headline — 150-step run (job 8540864)

- 150/150 steps logged, exit=0
- `peak_memory_active` ~24 GiB constant; `peak_memory_reserved` 41.9 → 58.8 GiB
  (climbs once early, then stable from ~step 70 onward)
- `ratios=1.0` invariant preserved throughout (single-epoch on-policy)
- One transient `kl_loss` spike at step 55 (6.73), settled by step 70 —
  bounded by `kl_coeff=0.02` in the gradient
- `grad_norm` healthy (mean 109–177 across the run, no NaN)
- LR schedule converged: 5e-7 → 5e-6 by step 10, cosine-decayed to 4.996e-6 by step 150
- Reward dynamics: rew_mean ~0.10, succ_mean ~0.025, response_lengths **stuck at 511.0**
  — this is the bug Stage 1 fixes.

## Stage 1 fix — vLLM stop strings + EOS injection (2026-06-13)

### Diagnosis

A direct probe of the vLLM HTTP server against the AGPT-2B local copy:

```
POST /v1/completions {prompt: "The capital of France is", max_tokens: 50, temperature: 0.0}
→ finish_reason: "length", text: " Paris.\n\nThe capital of France is Paris.\n\n... (repeating)"
```

AGPT-2B's `config.json` has `eos_token_id: 2`, but the raw pretraining
checkpoint **never naturally emits that token** — it generates fluent
rambling until `max_tokens`. The tokenizer's `stop_tokens = [eos_id]` was
being assembled correctly in the recipe; the recipe just had no way to
forward it (or anything else) to vLLM.

Inside the model the prompt template's natural format-end markers (`</answer>`,
the conversational turn `User:`) ARE produced by the model — see the 150-step
run's `SAMPLE_RESPONSE` logs, which show `</think>` and `</answer>` tags
across many sequences. But because vLLM didn't know to stop on them either,
the model rambled past them into hallucinated multi-turn conversations
(`</answer> User: ... Assistant: ...`) all the way to 511 tokens. The
`FormattedMathCorrectnessReward` parser sometimes saw garbage past `</answer>`
that polluted the partial-credit branch (`answer in completion` heuristic),
inflating the apparent baseline reward.

### Fix

Three coordinated changes:

1. **`torchtune/dev/rl/vllm_client.py`** — added `stop_token_ids` and `stop`
   (string list) parameters to `VLLMClient.generate` / `_generate_openai` /
   `_generate_trl`. When `stop` is non-empty, also passes
   `include_stop_str_in_output: true` so downstream regex-based reward
   extractors still see `</answer>` in the completion text.
2. **`recipes/dev/grpo_full_finetune_distributed_xpu.py`** — `_call_vllm_http`
   now forwards `self._stop_token_ids.cpu().tolist()` and a new
   `self._stop_strings` field (from `cfg.stop_strings`) to vLLM. The recipe
   also writes `eos_id` at the first pad position after each returned
   completion, so `truncate_sequence_at_first_stop_token` finds the boundary
   and `response_lengths` / `num_stop_tokens` reflect the actual generation.
3. **`recipes/configs/dev/production/auroragpt_2b_grpo_2n_gsm8k_xpu.yaml`** —
   added:
   ```yaml
   stop_strings:
     - "</answer>"
     - "User:"
     - "\nUser:"
   ```

### Validation (50-step bisect + full 150-step production run)

| Metric                          | Pre-fix 150 (8540864) | Stage 1c 50 (8541808) | **Stage 1 production 150 (8541918)** |
|---------------------------------|-----------------------|------------------------|--------------------------------------|
| Steps clean / submitted          | 150 / 150 | 50 / 50 | **150 / 150** |
| `num_stop_tokens` mean / 16     | **0.0** | **14.7** | **14.44** |
| Steps with `num_stop_tokens > 0` | 0 / 150 | 50 / 50 | **150 / 150** |
| `response_lengths` mean         | **511.0** | **111** | **121** |
| `rew_mean` overall              | 0.108 | 0.080 | 0.081 |
| `succ_mean` overall             | 0.025 | 0.026 | 0.028 |
| `ratios` invariant              | 1.0 | 1.0 | 1.0 |
| `peak_memory_active`            | ~24 GiB flat | ~24 GiB flat | ~24 GiB flat |
| `peak_memory_reserved`          | 41.9 → 58.8 GiB stable | 41.9 → 47.4 GiB stable | 47.4 → 58.8 GiB stable |
| Per-25-step rew_max             | 0.64 / 0.50 / 0.53 / 0.44 / 0.47 / 0.41 | n/a (50 step) | 0.50 / 0.53 / 0.53 / 0.44 / 0.31 / 0.50 |
| Per-25-step succ_max            | 0.44 / 0.19 / 0.19 / 0.25 / 0.19 / 0.13 | n/a | 0.19 / 0.31 / 0.12 / 0.25 / 0.06 / 0.12 |
| KL bucket peak                  | 2.13 (step 60 spike) | 0.71 | 0.59 (steps 51-75) |
| Wall                            | ~32 min | ~11 min | ~33 min |
| Exit                            | 0 | 0 | **0** |

The bucket rew_max and succ_max values are comparable to the pre-fix run
(the pre-fix run's higher rew_mean partly reflected partial-credit hits from
rambling text containing the gold answer by chance — a degenerate signal).
Post-Stage-1, the model must actually emit a parseable answer to score. This
is the right semantics for GRPO.

## Fixes that got us here

This run's stability inherits a chain of prior fixes; the full failure tree
lives in `docs/reports/agpt2b_2n_gsm8k_step62_wall_20260612.md`. Short list:

1. **Llama Q/K un-permute on weight sync** (`torchtune/dev/rl/weight_sync.py`,
   2026-06-11) — AGPT-2B's `LlamaForCausalLM` Q/K weights have a different
   layout than vLLM expects; the sync was sending scrambled weights, making
   vLLM emit pretraining noise after every sync. Fix logs
   `QK un-permute ENGAGED (n_heads=16 n_kv_heads=4 head_dim=128)` at first
   sync — the smoke test asserts this line.
2. **`TORCHTUNE_USE_CHUNKED_LOSS=0` (= chunked train backward)** —
   the single-backward path triggered a deterministic step-62 SIGABRT from
   L0 collective stall on the policy backward's allgather; chunked at
   `forward_batch_size=8` clears the wall (see CLAUDE.md `feedback_torchtune_use_chunked_loss_is_inverted.md` —
   the flag name is inverted from what it does).
3. **`WSYNC_CROSS_METHOD=gloo`** — cross-node XCCL/RDMA wsync hit the CXI MR
   cache leak (`memory/project_gloo_cross_pg_fix.md`); gloo TCP over hsn0
   is bandwidth-equivalent for AGPT-2B's payload size and has no leak.
4. **`TORCHTUNE_VARLEN_NOGRAD_BYPASS=0`** — bypass mask on the no-grad ref
   forward sent unsafely-masked logprobs to the loss when the ref is not
   task-converged (raw pretraining + GSM8K = AGPT-2B's exact case). With
   bypass on, `kl_loss` exploded 6.7e7 → inf by step 36; off, it stays
   bounded under `kl_coeff=0.02`. See
   `memory/feedback_varlen_nograd_bypass_unsafe_on_unconverged_ref.md`.
5. **Stable hparams**: `lr=5e-6` (vs 1e-5 baseline that NaN'd by step 40),
   `kl_coeff=0.02` (vs 0.0), `num_warmup_steps=10` (vs 5).

## Known gaps / next levers

- **Reward trend is not yet strongly monotone in 50 steps.** Per-step
  success rates are 0–0.19; bucket means hover around 0.025–0.044. The model
  produces formatted answers reliably (`<answer>...</answer>` in most
  completions) but the math is frequently wrong, and a non-trivial fraction
  of steps degenerate into tag-spam loops. Next levers to try, in order of
  expected effort:
  - **Stage 2**: `always_compute_rollout_logprobs: true` to give the policy a
    real `pi_old` to compare against (currently `ratios=1.0` constant, so
    GRPOSimpleLoss's clip term is a no-op). Cheapest config flip.
  - **Stage 3**: bump `ThinkingAnswerFormattingReward.positive_reward: 0.1 → 0.3`
    so well-formatted answers get more credit relative to (rare) math wins.
    Config-only change.
  - **Stage 4**: extend `max_generated_tokens: 512 → 1024` after the previous
    levers land. Stretch goal — could reawaken the sig#2 wall.
- **AGPT-2B is a raw pretraining checkpoint.** Even with stop_strings, the
  base model has no fine-tuned tendency to terminate at `</answer>`; it just
  doesn't emit it after `<answer>X` half the time. An SFT pass on the GSM8K
  format would likely transform learning. Out of scope for this report —
  this exemplar establishes the engine.
- **Per-step SAMPLE_RESPONSE logs sequence 0 only.** The aggregate stats line
  is the source of truth for reward / success rates across the 16-sample group.

## Repro

```bash
# Smoke test (5 steps, ~10 min, debug queue)
qsub experiments/auroragpt_2b_bakeoff/smoke_2n_gsm8k.sh

# Full production run (150 steps, ~32 min training + ~5 min vLLM startup)
qsub experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh
```

The smoke asserts the invariants that protect the production envelope; the
production launcher runs the validated 150-step path. Both go via the same
underlying recipe and config.
