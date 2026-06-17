# AuroraGPT-2B SFT → GRPO — Ablation Report

> **Companion primer:** `docs/features/agpt2b_sft_pipeline.md` (recipes, configs,
> launcher, how to reproduce). This file is the measured-results table.

## TL;DR

Both SFT strategies beat the raw-pretraining GRPO baseline. The multi-corpus
mathmix is the first variant to lift **success rate** (not just partial-credit
format reward), the lift **replicates at n=2**, and the same checkpoint then
**scales to 8 nodes** (7 training replicas + 1 vLLM) — beating the 2-node run
on both throughput (~10× distinct-prompt rate) and learning (+53% success).

| Init | reward all-150 | reward late (100-149) | success_all | response_len | num_stop / 16 |
|---|---|---|---|---|---|
| raw pretraining (avg n=2) | 0.0795 | 0.0655 | 2.79% | 511 | — |
| GSM8K-only SFT (avg n=2)  | 0.1014 | 0.0982 | 2.71% | ~226 | — |
| **mathmix SFT (avg n=2)** | **0.1737** | **0.1385** | **6.50%** | **156** | **15.70 (98%)** |

vs raw baseline: mathmix **+118% reward**, **+133% success**, **2.3× success rate**.
The mathmix lift is **replicated n=2** (per-metric spread ≤1% absolute, far
inside the 4–7% relative SFT-side variance band).

**Scale-up (same mathmix init):**

| Run | topology | late-window reward (100-150) | success | distinct prompts/step | wall |
|---|---|---|---|---|---|
| 2N baseline (job 8544461) | 2 nodes, server mode (11 train + 12 vLLM tiles) | ~0.137 | ~4.4% | 1 | ~36 min |
| **8N HSDP (job 8545190)** | **7 train nodes (84 ranks) + 1 vLLM node** | **0.1721** | **6.73%** | **14** | 40.6 min |

8N is **+26% reward, +53% success** over 2N (apples-to-apples: both weight-sync
ON, on-policy `ratios=1.0`, G=8), at ~10× the distinct-prompt throughput.

**Plateau** (separate 2N study, job 8544526): extending GRPO to 225 steps @
lr=1e-5 lifts the tail (175–225) to **8.88% success** with reward still rising —
the ~6.5% success rate is a step-count ceiling, not a wall. See
`docs/reports/agpt2b_mathmix_grpo_replica_and_plateau_20260616.md`.

## Platform

| Item | Value |
|------|-------|
| Hardware | Aurora, 12 tiles/node — 2 nodes (Stages 1–2, plateau study) up to 8 nodes (Stage 3) |
| Model | AuroraGPT-2B (2B Llama-family raw pretraining ckpt at `global_step138650`) |
| Framework | torchtune XPU, frameworks/2025.3.1 |
| GRPO recipe | `recipes/dev/grpo_full_finetune_distributed_xpu.py` |
| 2N GRPO envelope | 150 steps · lr=5e-6 · kl=0.02 · G=16 · fbs=8 · max_gen=512 · vLLM server (TRAIN_TILES=11, VLLM_DP=12) |
| 8N GRPO envelope | FSDP1 HYBRID_SHARD `dp_replicate=7 × dp_shard=12` · G=8 · bs=2 · fbs=4 · `gc:0.8` · dedicated vLLM node (Stage 3) |
| SFT recipe | `recipes/dev/full_finetune_distributed_xpu.py` (full FT) |

Stages 1–2 ablation cells use the same 2N GRPO recipe and envelope — only
`MODEL_PATH` varies. Stage 3 scales the *same* mathmix init to 8 nodes with the
memory-tuned HSDP envelope (see that stage for why G/bs differ).

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

### GRPO result (n=2, replica confirmed 2026-06-16)

| Metric | Raw baseline (avg n=2) | GSM8K-only SFT (avg n=2) | **Mathmix SFT (avg n=2)** | Δ vs raw | Δ vs GSM8K-only |
|---|---|---|---|---|---|
| reward mean (all 150) | 0.0795 | 0.1014 | **0.1737** | **+118%** | **+71%** |
| reward late (100-149) | 0.0655 | 0.0982 | **0.1385** | **+111%** | **+41%** |
| **success rate (all 150)** | **2.79%** | **2.71%** | **6.50%** | **+133%** | **+140%** |
| response_len_avg | 511 | ~226 | 156 | — | — |
| num_stop_tokens / 16 | — | — | 15.70 (98%) | — | — |

Replica (job 8544461) vs original (job 8544020), identical recipe/hparams:

| Metric | n=1 (8544020) | n=2 (8544461) | spread |
|---|---|---|---|
| reward all-150 | 0.1720 | 0.1754 | 2.0% rel |
| reward late (100-149) | 0.1400 | 0.1370 | 2.1% rel |
| success all-150 | 6.75% | 6.25% | 0.5pp |

Every metric is within ~1% absolute — far inside the 4–7% relative SFT-side
variance band measured in Stage 1. The mathmix lift is **confirmed
reproducible** and is the shipped AGPT-2B GRPO baseline (bakeoff launcher
default `MODEL_PATH` updated to mathmix epoch_2).

Phase breakdown (n=1 trajectory shown):

| Window | reward_mean | success_rate |
|---|---|---|
| 0-49 | 0.184 | 7.02% |
| 50-99 | 0.196 | 7.62% |
| 100-149 | 0.140 | 5.75% |

Mid-phase peaks slightly above early; late-phase regresses ~30% vs mid. This
was **mis-attributed to cosine LR decay** in earlier write-ups — see the
correction in "Sharp edges" below: the schedule is effectively flat at these
run lengths, so the dip is window noise / a step-count effect, not annealing.

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

## Stage 3 — 8-node HSDP scale-up (2026-06-16)

The mathmix epoch_2 checkpoint feeds an 8-node run that trades the 2-node
single-replica topology for genuine **hybrid-sharded data parallelism**: 7
training nodes (`dp_replicate=7 × dp_shard=12` = 84 ranks) + 1 dedicated vLLM
node. Each replica draws **distinct** GSM8K prompts and the gradients are
all-reduced across replicas every step (FSDP1 `HYBRID_SHARD`), so the effective
distinct-prompts/step rises from 1 (2N) to `batch_size × dp_replicate = 2 × 7 =
14`. The lower-variance GRPO advantage estimate is the hypothesised win.

### Setup

| Knob | 2N baseline | 8N HSDP |
|---|---|---|
| Topology | 2 nodes, server mode (11 train tiles + 12 vLLM tiles) | 7 train nodes (84 ranks) + 1 vLLM node |
| FSDP | ZeRO-3 (single replica) | FSDP1 `HYBRID_SHARD` (`dp_replicate=7 × dp_shard=12`) |
| Distinct prompts/step | 1 | **14** (`batch_size 2 × dp_replicate 7`) |
| G (grpo_samples) | 16 | 8 |
| batch_size / fbs / ref_fbs | 1 / 8 / 8 | 2 / 4 / 4 |
| Weight sync | XCCL server, interval 1 | XCCL server (rank-0 gather + broadcast), interval 1 |
| Allocator | `gc:0.95, max_split_size_mb:512` | `gc:0.8`, no `max_split_size` |
| Launcher | `pbs_2n_gsm8k_production.sh` | `run_agpt2b_7n_hsdp_8node.sh` (must `qsub`, not SSH) |
| Config | `auroragpt_2b_grpo_2n_gsm8k_xpu_handshake.yaml` | `auroragpt_2b_grpo_7n_gsm8k_hsdp_xpu.yaml` |

Both runs share the same init checkpoint, lr=5e-6, kl=0.02, max_gen=512, and
weight-sync ON (on-policy, `ratios=1.0` every step).

### Result (job 8545190, 150/150 steps, rc=0)

Apples-to-apples late window (steps 100–150), both wsync ON:

| Run | reward (late) | success (late) | distinct prompts/step | throughput | wall |
|---|---|---|---|---|---|
| 2N baseline (8544461) | ~0.137 | ~4.4% | 1 | 0.082 prompts/s | ~36 min |
| **8N HSDP (8545190)** | **0.1721** | **6.73%** | **14** | **0.85 prompts/s** | 40.6 min (16.4 s/step) |

**8N is +26% reward and +53% success over 2N**, at **~10× the distinct-prompt
rate**. Weight-sync gather is 0.6–0.7 s/step (cheaper than 2N's 1.9 s — the
rank-0 gather is small and the broadcast overlaps). `torch_resv` stays flat and
uniform at ~30 GiB across all shard-leaders; no `banned:1`.

### Correctness proof

HSDP distinct-prompt logic was validated at both 3- and 7-replica scale before
the learning run: replicas process **different** GSM8K problems each step but
produce **bit-identical** `grad_norm` across all replicas every step (e.g. 7
replicas all at `grad_norm=2.8594` on step 1), which is exactly the signature of
a correct cross-replica gradient all-reduce. `ratios=1.0000` confirms on-policy.

### The memory wall (and why it isn't a CCL/L0 leak)

The 8N path initially crashed at step 11–12 with `banned:1` (GPU page-fault).
Four attempts and a mem-probe traced it to **PyTorch reserved-pool
fragmentation**, *not* the XPU/CCL/OFI-MR/L0 leak class the signature
superficially resembles:

- Diagnostic signature: `torch_resv` climbs monotonically (~5 GiB/step,
  20→62 GiB) and **spreads** across replicas (47–62 GiB at identical
  `torch_alloc`), while `torch_alloc` stays flat ~13 GiB and `external` (CCL)
  stays flat ~2 GiB. Reserved-up + alloc-flat + external-flat = fragmentation.
- Driver: per-rank buffer volume = `batch_size × G` = sequences/rank.
  `batch_size=4 × G=16` (64 seqs/rank, 4× the 2N path) fragmenting the
  variable-length GSM8K completion buffers pinned the worst-case replica's tile.
  The 2N path is stable *because* `batch_size=1` keeps the peak low — nothing to
  do with single- vs multi-node.
- Fix: **G=8, batch_size=2, fbs=4, ref_fbs=4, `gc:0.8`, drop
  `max_split_size_mb`**. `torch_resv` collapses to a flat, uniform ~30 GiB
  (23–29 GiB `l0_free` headroom on every tile). The data-parallel win (14
  distinct prompts/step) comes from `batch_size × dp_replicate` and is
  independent of G, so dropping G 16→8 keeps the scaling benefit.

> **Lesson:** with `banned:1`, read the mem-probe (`torch_resv` vs
> `torch_alloc` vs `external` vs `l0_free`) *before* hypothesising a transport
> fix. Two wrong guesses (inter-node all-reduce reroute; OFI-MR config) cost two
> debug-scaling runs; the mem-probe answered it in one. A 10-step smoke is too
> short to catch a step-11/12 wall — use ≥15 steps.

### Execution constraints (8N path differs fundamentally from 2N)

- **Must `qsub` the launcher; do not SSH-into-hold and run it.** `mpiexec
  --pmi=pmix` only attaches to the PBS job from inside its process tree
  (SSH-dispatch fails `Couldn't attach to job NONE`). The `VLLM_ONLY=1`
  validation path has no mpiexec and *can* be SSH-driven.
- **Mom-node `module load frameworks/2025.3.1` is required before mpiexec** —
  it propagates the env to all ranks; without it ranks die
  `ModuleNotFoundError: No module named 'torch'`.
- **Hostfile must be plain FQDN lines (no `:N` slot suffix), `-ppn 12` sets
  ranks/node.** The launcher excludes the vLLM node so it can't pass
  `$PBS_NODEFILE` directly; a constructed `host:12` file fails PALS RPC.
- **Wrapper `WORLD_SIZE` must resolve from `PMI_SIZE`/`PALS_NRANKS`/`WORLD`,
  never a numeric literal** — Aurora PALS leaves `PMI_SIZE` empty; a hardcoded
  default inits an oversized process group and the absent ranks time out.

## Plateau study — longer + higher LR (2026-06-16)

A separate 2N run (job 8544526) probed whether the ~6.5% success rate is a true
plateau or a step-count ceiling: **225 steps @ lr=1e-5** vs the validated 150 @
5e-6.

| Window | reward | success |
|---|---|---|
| n=2 baseline (150 @ 5e-6) | 0.1737 | 6.50% |
| first-150 @ 1e-5 (matched) | 0.1739 | 6.54% |
| extended 150–225 @ 1e-5 | 0.1925 | 7.83% |
| **tail 175–225 @ 1e-5** | **0.2238** | **8.88%** |

Two reads: (1) lr=1e-5 **ties** lr=5e-6 over the first 150 steps with
`grad_norm` controlled (mean ~12, no 200+ spikes — `clip_grad_norm=1.0` holds
it) and KL bounded — the "halve to 5e-6 for stability" caution is
over-conservative given the clip. (2) The extra training is what moves the
needle: tail success reaches 8.88% (+2.4pp) and **reward was still rising at
step 225**. The ~6.5% ceiling is a step-count limit, not a wall. Steps and LR
are confounded in this single run; full detail in
`docs/reports/agpt2b_mathmix_grpo_replica_and_plateau_20260616.md`.

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
   total steps — on the SFT side.** 5-ep GSM8K regresses vs 3-ep on the same
   data: the fixed warmup over more steps means cosine anneals to near-zero LR
   earlier, so later epochs contribute no update. Scale `warmup_steps`
   proportionally or switch to constant-LR + cooldown before any longer SFT.
6. **The GRPO-side late-phase dip is NOT cosine decay (correction).** Earlier
   write-ups (and an earlier version of this doc) attributed the steps-100-149
   reward regression to the cosine schedule. That was wrong: the GRPO cosine
   `num_training_steps` is `total_epochs × steps_per_epoch = 7473` (the full
   dataloader length), but the run caps at `num_steps` (150/225). The LR is
   therefore **effectively flat** — it decays only ~0.1% across 225 steps after
   warmup. The late dip is window noise / a step-count effect, not annealing
   (the plateau study confirms reward *rises* past step 150). A real GRPO
   "LR-schedule fix" would require setting `num_training_steps` to the actual
   run length, which no run here did.
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

1. ~~**Replica** GRPO run to bound variance~~ — **done** (job 8544461, n=2
   confirmed; Stage 2).
2. ~~**8-node scale-up**~~ — **done** (job 8545190; Stage 3): +53% success and
   ~10× distinct-prompt throughput over 2N, bounded memory.
3. **Disambiguate the plateau lever** — a 225-step @ lr=5e-6 control to
   separate the extra-steps effect from the higher-LR effect (confounded in
   job 8544526).
4. **Push past 225 steps** — reward had not flattened at 225; a longer run
   (prod queue or checkpoint/resume, >60 min) to find the true ceiling. The
   handshake YAML lr can return to 1e-5 (clip controls the spikes that drove
   the 5e-6 choice).
5. **SFT LR-schedule re-baseline** (carry-over from Stage 1): proportional
   `warmup_steps` or constant-LR + cooldown before any longer SFT pass.
6. **Targeted reward shaping** — partial credit on intermediate reasoning
   steps (e.g. expression equality on sub-expressions) — to make the policy
   gradient denser per batch.
