# AuroraGPT-2B SFT → GRPO — Primer

> **Status (2026-06-15):** Both phases land. GSM8K-only SFT validated end-to-end
> 2026-06-14 (+28% mean GRPO reward over raw init). Multi-corpus mathmix SFT
> validated 2026-06-15 (success rate 2.71% → 6.75%; n=1, replica pending).
> See `docs/experiments/agpt2b_sft_to_grpo.md` for the full ablation.

## Why this exists

AuroraGPT-2B is a 2B Llama-family raw pretraining checkpoint
(`/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650`).
It has never seen instruction data and does not naturally emit EOS — point it
at the GRPO production envelope and every completion runs to
`max_generated_tokens` with no learning signal.

Two things have to happen before RL works:

1. **Format alignment** — teach the model to emit
   `<think>...</think> <answer>...</answer>`, the template the GRPO reward
   parser (`FormattedMathCorrectnessReward`) actually scores against.
2. **Capability lift** — give the model enough exposure to math reasoning
   that some non-trivial fraction of rollouts are fully correct, so the
   policy gradient is not sparsity-bound.

GSM8K-only SFT handles (1) and partly (2). The multi-corpus mathmix handles
both simultaneously.

## The pipeline

```
Raw pretraining ckpt  ──► SFT (full FT, ZeRO-3)  ──► GRPO (vLLM server mode)
  global_step138650        epoch_2/model.safetensors    pbs_2n_gsm8k_production.sh
```

The same launcher (`experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh`)
runs the GRPO stage for any of the SFT outputs — just override `MODEL_PATH`.

## What landed

### Recipes

- `recipes/dev/full_finetune_distributed_xpu.py` —
  `FullFinetuneRecipeDistributedXPU`. Standard full FT recipe with a
  top-of-module `sys.modules['torchtune']` + `install_xpu_patches()` shim so
  XCCL-sensitive imports are pinned before any torchtune submodule loads.
- `recipes/dev/lora_finetune_distributed_xpu.py` — LoRA variant, same shim.

Both recipes import on a no-XPU host (verified by the CPU regression test).

### Tokenizer

- `torchtune.dev.rl.ezpz_tasks.AuroraGPTTokenizer` — added
  `tokenize_messages(messages, *, add_end_tokens=True)` and `__call__(sample,
  inference=False)`. AGPT-2B has no chat template; the tokenizer emits plain
  text concatenation `<bos> {user} {assistant} <eos>` with role-correct
  masking, so the standard `SFTDataset` / `SFTTransform` can consume it
  without a Jinja template.

### Datasets

- `torchtune.dev.sft.auroragpt_math_mix.auroragpt_math_mix_sft(...)` —
  `ConcatDataset` of GSM8K + MATH + MetaMathQA. Every sample is rendered into

  ```
  PREAMBLE_PROMPT.format(question=q) + "<think>{cot}</think> <answer>{ans}</answer>"
  ```

  i.e. the exact format the GRPO reward parser expects. Defaults:
  `gsm8k_replicas=3, math_replicas=1, metamath_replicas=1, metamath_subset=25000`
  → ~60K rows/epoch.

  **Truncation guard:** the loader hard-caps every sample at
  `tokenizer.max_seq_len`, dropping preamble tokens first (preamble is
  loss-masked anyway). A 200-row smoke missed the MATH long tail (p99=1356,
  max=2758); full-corpus sweep hits the 2048 cap on 15/59,917 rows (0.025%).

### Configs

All under `recipes/configs/dev/production/`:

| Config | Purpose |
|---|---|
| `auroragpt_2b_sft_alpaca_smoke_xpu.yaml` | ~30-step Alpaca smoke (full FT) |
| `auroragpt_2b_sft_alpaca_smoke_lora_xpu.yaml` | Alpaca smoke (LoRA) |
| `auroragpt_2b_sft_gsm8k_xpu.yaml` | **GSM8K-only SFT winner** (3-ep lr=1e-5) |
| `auroragpt_2b_sft_gsm8k_xpu_safe.yaml` | GSM8K-only, `compile: false` fallback |
| `auroragpt_2b_sft_gsm8k_lr5e6_xpu.yaml` | GSM8K-only, lr=5e-6 ablation arm |
| `auroragpt_2b_sft_gsm8k_5ep_xpu.yaml` | GSM8K-only, 5-epoch ablation arm |
| `auroragpt_2b_sft_lora_gsm8k_xpu.yaml` | GSM8K-only (LoRA) |
| `auroragpt_2b_sft_mathmix_xpu.yaml` | **Mathmix SFT winner** (3-ep lr=1e-5) |
| `auroragpt_2b_grpo_2n_gsm8k_xpu_handshake.yaml` | GRPO sidecar: same as the prod `..._xpu.yaml` except `checkpoint_files: [model.safetensors]` (SFT recipe writes safetensors, not `pytorch_model.bin`) |

The mathmix config forces `compile: false`. Variable-length math corpora
exhaust `torch._dynamo` `recompile_limit (8)` by per-shape recompile within
the first epoch — backbone-only compile is incompatible without
`packed=true` or shape bucketing.

### Launcher default

`experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh` switched from

```bash
export CONFIG=...
export MODEL_PATH=...
```

to

```bash
export CONFIG=${CONFIG:-recipes/configs/.../auroragpt_2b_grpo_2n_gsm8k_xpu_handshake.yaml}
export MODEL_PATH=${MODEL_PATH:-/lus/flare/.../gsm8k_2n_full_20260614_041943/run_out/epoch_2}
```

so parent launchers can override. The default points to the 2026-06-14
GSM8K-only SFT epoch_2; to reproduce the raw-pretraining baseline:

```bash
MODEL_PATH=/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650 \
  CONFIG=recipes/configs/dev/production/auroragpt_2b_grpo_2n_gsm8k_xpu.yaml \
  qsub experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh
```

To use the mathmix SFT instead:

```bash
MODEL_PATH=/lus/flare/.../mathmix_2n_full_20260615_043457/run_out/epoch_2 \
  qsub experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh
```

### Tests

`tests/torchtune/dev/sft/test_sft_xpu_import.py` (CPU-safe, ~7s on a login
node):

- Pins both XPU SFT recipes import on a no-XPU host (validates the
  `sys.modules['torchtune']` shim survives without an XPU device).
- Pins `AuroraGPTTokenizer` exposes `tokenize_messages` + `__call__`
  (regression — the GRPO custom dataset bypassed the standard SFT contract
  and the tokenizer used to lack the methods entirely).

Run:

```bash
module load frameworks
pytest tests/torchtune/dev/sft --timeout=60 -v
```

## Reproducing a fresh end-to-end run

1. **SFT** (2N, ~30 min for GSM8K-only, ~3h for mathmix on capacity queue):

   ```bash
   # GSM8K-only
   tune run --nproc-per-node 12 full_finetune_distributed_xpu \
     --config recipes/configs/dev/production/auroragpt_2b_sft_gsm8k_xpu.yaml

   # OR mathmix
   tune run --nproc-per-node 12 full_finetune_distributed_xpu \
     --config recipes/configs/dev/production/auroragpt_2b_sft_mathmix_xpu.yaml \
     compile=false
   ```

   Output: 3 checkpoint dirs `run_out/epoch_{0,1,2}/model.safetensors` (3.8–4.0 GB each).

2. **GRPO** (2N, ~36 min for 150 steps):

   ```bash
   MODEL_PATH=<your SFT run>/run_out/epoch_2 \
     qsub experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_production.sh
   ```

## Known sharp edges

- **`compile: false` is required for any variable-length corpus.** The
  GSM8K-only +12-23% compile win depends on GSM8K's tight shape envelope
  (max=606). Mathmix has MATH p99=1356 and exhausts dynamo's recompile limit
  in the first epoch.
- **Cosine LR schedule + small fixed `warmup_steps=20` becomes a trap for
  longer SFT runs.** 5-ep regresses vs 3-ep on the same data (warmup is a
  much smaller fraction of total steps, so late epochs run at near-zero LR).
  Scale `warmup_steps` proportionally or switch to constant + cooldown
  before adding more epochs.
- **`TORCHTUNE_VARLEN_NOGRAD_BYPASS=1` is UNSAFE on the raw AGPT-2B + GSM8K
  ref**. The bypass drops the explicit causal+padding mask on no-grad
  forwards; this is bit-exact safe when the ref is task-converged, but on
  un-converged refs the KL term goes to NaN by step ~40. The default
  launcher leaves it OFF.
