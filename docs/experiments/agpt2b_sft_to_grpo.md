# AuroraGPT-2B SFT → GRPO — Ablation Report

> **Companion primer:** `docs/features/agpt2b_sft_pipeline.md` (recipes, configs,
> launcher, how to reproduce). This file is the measured-results table.

## TL;DR

Both SFT strategies beat the raw-pretraining GRPO baseline. The multi-corpus
mathmix is the first variant to lift **success rate** (not just partial-credit
format reward).

| Init | reward all-150 | reward late (100-149) | success_all | response_len | num_stop / 16 |
|---|---|---|---|---|---|
| raw pretraining (avg n=2) | 0.0795 | 0.0655 | 2.79% | 511 | — |
| GSM8K-only SFT (avg n=2)  | 0.1014 | 0.0982 | 2.71% | ~226 | — |
| **mathmix SFT (n=1)**     | **0.1720** | **0.1400** | **6.75%** | **155** | **15.68 (98%)** |

vs raw baseline: mathmix **+116% reward**, **+142% success**, **2.4× success rate**.

n=1 vs n=2 caveat — formal replica is pending; SFT-side variance band on
GSM8K-only was 4–7% relative and the mathmix lift is far outside that.

## Platform

| Item | Value |
|------|-------|
| Hardware | 2 × Aurora nodes, 12 tiles/node, 24 ranks total |
| Model | AuroraGPT-2B (2B Llama-family raw pretraining ckpt at `global_step138650`) |
| Framework | torchtune XPU, frameworks/2025.3.1 |
| GRPO recipe | `recipes/dev/grpo_full_finetune_distributed_xpu.py` |
| GRPO envelope | 150 steps · lr=5e-6 · kl=0.02 · G=16 · fbs=8 · max_gen=512 · vLLM server (TRAIN_TILES=11, VLLM_DP=12) |
| SFT recipe | `recipes/dev/full_finetune_distributed_xpu.py` (full FT) |

All ablation cells use the same GRPO recipe and envelope — only `MODEL_PATH`
varies.

## Stage 1 — GSM8K-only SFT (2026-06-14)

### Setup

| Knob | Value |
|---|---|
| Dataset | `torchtune.dev.rl.ezpz_tasks.gsm8k_instruct` (7,473 train rows) |
| Epochs × opt steps | 3 × 19 = 57 |
| Batch | bs=4, grad_accum=4 → effective 384 across 24 ranks |
| Optimizer | AdamW lr=1e-5, cosine, warmup=20 |
| Stack | LinearCE, packed=false, AC=true, ZeRO-3, bf16 |
| Compile | backbone-only via DictConfig (`model: true`, others false) |
| Wall | 28 min (rc=0) |
| Loss | 6.40 → 0.78 |

`compile: true` scalar SIGSEGVs on XPU; the DictConfig form works and is the
+12-23% lift validated in the 1N tuning pass (memory:
`project_agpt2b_sft_aurora_tuning_20260614`).

### GRPO ablation (150 steps, paired replicas both sides)

| SFT config | final SFT loss | mean reward | late (100-149) | success | Δ vs raw avg |
|---|---|---|---|---|---|
| raw baseline (run 1) | — | 0.0810 | 0.0646 | 2.79% | — |
| raw baseline (run 2) | — | 0.0779 | 0.0660 | 2.79% | — |
| **raw baseline (avg n=2)** | — | **0.0795** | **0.0655** | **2.79%** | (ref) |
| 3-ep, lr=1e-5 (run 1) | 0.78 | 0.1050 | 0.1050 | 2.88% | +32% |
| 3-ep, lr=1e-5 (run 2) | 0.78 | 0.0979 | 0.0914 | 2.54% | +23% |
| **3-ep, lr=1e-5 (avg n=2)** | — | **0.1014** | **0.0982** | 2.71% | **+28%** |
| 3-ep, lr=5e-6 (single) | 0.88 | 0.0998 | 0.0950 | 2.50% | +26% |
| 5-ep, lr=1e-5 (single) | 0.87 | 0.0940 | 0.0764 | 1.92% | +18% |

**Sweet spot: 3 epochs at lr=1e-5.**

- 5-epoch regresses: fixed `warmup_steps=20` over 5 epochs (95 steps) means
  cosine anneals to near-zero LR much earlier; later epochs contribute no
  parameter update. Late-phase late-LR overtraining also drops policy entropy
  (success 2.88% → 1.92%).
- lr=5e-6 SFT (gentler) hypothesis — that preserving more entropy would help
  RL exploration — was not borne out. It lands within noise of lr=1e-5 on
  mean reward but lower on success (2.50% vs 2.88%).

### Variance — measured both sides

- Raw-side spread (n=2): 0.0810 vs 0.0779 — 4% relative.
- SFT-side spread (n=2): 0.1050 vs 0.0979 — 7% relative.
- The +28% averaged lift is far outside both bands.
- **Late-phase reward** (steps 100-149) is the more stable single-trajectory
  signal: raw-avg 0.0655 vs SFT-avg 0.0982 = **+50%**, real.

### The bottleneck this run flagged

Success rate barely moved (2.79% → 2.71%) despite reward going up. The lift
came entirely from the loose `answer in completion` partial-credit path of
the reward — the strict `FormattedMathCorrectnessReward` path needs
`<answer>...</answer>` tags, and the `instruct_dataset` SFT path produced
`The answer is 42` style completions, so the strict reward never fired.

## Stage 2 — Multi-corpus mathmix SFT (2026-06-15)

### Setup

| Source | Rows / epoch | Replicas | Effective | Why |
|---|---|---|---|---|
| GSM8K (`openai/gsm8k`) | 7,473 | × 3 | 22,419 | Format anchor / downstream eval target |
| MATH (`qwedsacf/competition_math`) | 12,498 (post `\boxed{}` filter) | × 1 | 12,498 | Harder competition problems |
| MetaMathQA subset (`meta-math/MetaMathQA`) | 25,000 (seeded random) | × 1 | 25,000 | Wide CoT diversity (rephrasings, FOBAR, SV) |
| **Total per epoch** | | | **59,917** | |

Per-token GSM8K share lands ≈ 25–30%. Every sample rendered as
`PREAMBLE_PROMPT.format(question=q) + "<think>{cot}</think> <answer>{ans}</answer>"`.

| Knob | Value |
|---|---|
| Epochs × opt steps | 3 × 156 = 468 |
| Batch | bs=4, grad_accum=4 → effective 384 |
| Optimizer | AdamW lr=1e-5, cosine, warmup=20 |
| Stack | LinearCE, packed=false, AC=true, ZeRO-3, bf16 |
| Compile | **off** (variable-length corpus exhausts dynamo recompile_limit) |
| max_seq_len | 2048 (truncate-preamble-first guard in loader) |
| Wall | 2h 50min (rc=0) |
| Loss | 7.03 → 0.58 |

### GRPO result (one run)

| Metric | Raw baseline (avg n=2) | GSM8K-only SFT (avg n=2) | **Mathmix SFT (n=1)** | Δ vs raw | Δ vs GSM8K-only |
|---|---|---|---|---|---|
| reward mean (all 150) | 0.0795 | 0.1014 | **0.1720** | **+116%** | **+70%** |
| reward late (100-149) | 0.0655 | 0.0982 | **0.1400** | **+114%** | **+43%** |
| **success rate (all 150)** | **2.79%** | **2.71%** | **6.75%** | **+142%** | **+149%** |
| response_len_avg | 511 | ~226 | 155 | — | — |
| num_stop_tokens / 16 | — | — | 15.68 (98%) | — | — |

Phase breakdown:

| Window | reward_mean | success_rate |
|---|---|---|
| 0-49 | 0.184 | 7.02% |
| 50-99 | 0.196 | 7.62% |
| 100-149 | 0.140 | 5.75% |

Mid-phase peaks slightly above early; late-phase regresses ~30% vs mid — the
same cosine + small-fixed-warmup artifact called out in Stage 1. The mathmix
init is starting from a higher base, so the regression is from a higher floor.

### Why this finally moved success rate

Two compounding effects:

1. **Format alignment** turns on the strict `FormattedMathCorrectnessReward`
   path. Under GSM8K-only SFT, even arithmetically-correct answers scored
   0.0 on the strict path because they weren't `<answer>...</answer>`-wrapped.
   Under mathmix, 98% of completions emit the tag correctly, so correct
   answers actually score 1.0.
2. **Corpus diversity** lifts the *underlying* solve rate. MATH and
   MetaMathQA expose the model to wider math reasoning patterns than GSM8K
   alone, so a wider distribution of GSM8K test prompts have at least one
   prior pattern the model can imitate.

The previous report's diagnosis ("the plateau is capability + reward sparsity,
not the loss component") holds — mathmix attacks both levers at once.

## Sharp edges (lessons that bit)

1. **`compile: true` is incompatible with variable-length corpora.** First
   mathmix attempt died step ~10 with `torch._dynamo recompile_limit (8)`.
   The GSM8K-only compile win depends on GSM8K's tight shape envelope
   (max=606); mathmix p99s are 520/1356/798. `compile: false` is the right
   default for any future multi-corpus SFT until packed/bucketed inputs are
   wired up.
2. **Dataset must hard-cap at `max_seq_len`, not just expect it from the
   loader.** Same first attempt also hit
   `seq_len (2800) > max_seq_len (2048)` because a 200-row smoke missed the
   MATH long tail (max=2758). The loader now truncates at tokenization time,
   dropping preamble first (loss-masked anyway). Full sweep: 15/59,917 hit
   the cap (0.025%) — fine.
3. **Cosine LR + small fixed `warmup_steps=20` becomes a trap as you grow
   total steps.** 5-ep GSM8K regresses vs 3-ep on the same data; both Stage
   1 and Stage 2 show a late-phase regression vs mid-phase. Scale
   `warmup_steps` proportionally or switch to constant-LR + cooldown before
   any longer SFT pass.
4. **`TORCHTUNE_VARLEN_NOGRAD_BYPASS=1` is unsafe on the raw AGPT-2B + GSM8K
   ref.** The bypass drops the explicit causal+padding mask on no-grad
   forwards; bit-exact safe on a task-converged ref (e.g. Qwen3-4B SFT'd),
   but on un-converged AGPT-2B raw + GSM8K, `kl_loss` goes 6.7e7 → inf and
   loss → NaN by step ~40. Default is OFF in the launcher. See memory
   `feedback_varlen_nograd_bypass_unsafe_on_unconverged_ref`.
5. **Launcher hardcoded `export CONFIG=` / `export MODEL_PATH=`** silently
   drops parent-launcher overrides. Wasted 2 runs early in this work. Fix
   landed: `${VAR:-default}` form. See memory
   `feedback_bakeoff_launcher_hardcodes_model_path`.

## Next steps

1. **Replica** GRPO run from the mathmix SFT init to bound variance
   (~36 min on debug queue).
2. **LR-schedule re-baseline** (carry-over from Stage 1): try `constant_lr`
   + short cooldown or proportional `warmup_steps` before any longer SFT.
3. **300-step GRPO from mathmix init** to see if the plateau still kicks in
   at the same step count, or if the higher capability ceiling shifts it
   later. Pair with the LR fix.
4. **Targeted reward shaping** — partial credit on intermediate reasoning
   steps (e.g. expression equality on sub-expressions) — to make the policy
   gradient denser per batch.
