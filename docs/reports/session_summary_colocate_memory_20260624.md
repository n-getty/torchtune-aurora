# Session report: LoRA-GRPO colocate memory work + mg1024 diagnosis (2026-06-24)

Top-level summary of the 2026-06-24 session. Goal going in: make Qwen3-4B / BioReason LoRA
colocate run at `max_generated_tokens=1024` on a 64 GiB Aurora tile (the paper/BioReason
envelope), motivated by TRL doing it on 40 GiB A100s. Outcome: two real memory fixes shipped;
the mg1024 *goal* is **blocked by a pre-existing non-deterministic vLLM-XPU driver fault**, now
isolated and documented — not achieved.

## What shipped to main (all opt-in / default-safe)

| Change | Commit | Status |
|--------|--------|--------|
| Bound the IPEX-varlen output-buffer cache (`attention_utils.py`) | `2d2accd5` | **Validated.** Killed a ~0.44 GiB/step live-memory leak in the no-grad ref forward; affects ALL variable-seqlen `TORCHTUNE_USE_IPEX_VARLEN=1` runs. A/B: varlen=0 → flat; bounded → flat + GREEN. |
| Chunked-vocab `LinearGRPOLoss` in LoRA recipe | `366dd037` | **Memory win validated** (training bwd peak −16.7 GiB). Opt-in by loss component; default `GRPOLoss` byte-identical. NOT an mg1024 production-path validation (see below). |
| Learning-regime + temperature fences | `0138ed9d`, `89ae68da` | Fail-fast on ppo_epochs>1 / async / compile / FSDP; temperature threaded so T≠1.0 configs work. |
| Base + BioReason fail-fast guards | `7278a1d9` (+ `b60fea56` repair) | LinearGRPOLoss refused where unsafe (base = FULL_SHARD+trained tied output; BioReason = HF backbone, no `skip_output_layer`). |
| Docs: memory-fix report + env-flag rows | `9be02599` | (contained an overclaim, corrected this session — see below) |
| Bug doc: non-deterministic colocate-gen page fault | `32efcd17` | The mg1024 blocker, fully characterized. |

CPU tests: `test_varlen_buffer_cache.py`, `test_linear_grpo_loss_equivalence.py` (incl.
temperature + gradient bit-equivalence), `test_linear_grpo_loss_recipe_guards.py`. Full RL CPU
suite green except the documented `bioreason_lora_peft` env-shadow failures.

## The mg1024 blocker (the headline negative result)

mg1024 LoRA colocate crashes ~step 7-8 with a `CCS NotPresent / PDE banned:1` GPU page fault
during **generation**, 24 GiB free (not OOM). After chasing — and refuting — five deterministic
explanations (co-tenancy, token-ceiling, weight-sync, KV-config, memory), the A/B that varied
one thing and got the **opposite of prediction** exposed the truth: **the fault is
non-deterministic** (same `max_gen=512` config crashed once and survived 9 steps back-to-back,
job 8559162). Corroborated by 2026-06-18 data (two `max_gen=384` runs: one survived, one
crashed). It is a probabilistic vLLM-on-XPU KV-paging fault, P(crash) rising with generation
volume — same XPU L0 driver-instability family as the documented `empty_cache`/UR + oneCCL
leaks. Full writeup: `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md`.

**Practical envelope:** max_gen ≤ 64 safe (AGPT-2B colocate production); ≤ 256 low-risk
(usually clears a budget, NOT a hard line); ≥ 384 flaky; 1024 blocked. Mitigations (all
heavyweight, none recipe-side): periodic vLLM engine respawn (W17/W19), checkpoint+auto-resume,
or server/dedicated vLLM mode.

## "Why Aurora OOMs where A100 doesn't" — corrected analysis

It was never a 40-vs-64 GiB fit problem. Four separable mechanisms, in order: (a) the
varlen-cache leak [fixed]; (b) single-vs-chunked backward [`USE_CHUNKED_LOSS=1`]; (c) FP32
full-vocab logits [LinearGRPOLoss]; (d) the non-deterministic generation page fault [the real
mg1024 blocker]. Also corrected: XPU *does* have a fused autograd FlashAttentionXPU bf16 kernel
(`mask=None`+`is_causal`); `TORCHTUNE_MASKFREE_CAUSAL=1` already routes the training forward
onto it at bs=1.

## BioReason port: decided NOT worth doing

BioReason's validated production path is **server / dedicated_rank mode at max_gen=1024** (works,
200-step baseline) — not colocate. LinearGRPOLoss is a colocate-specific memory optimization;
porting it to BioReason would (a) target a mode BioReason doesn't use, (b) require a separate
HF-backbone implementation (no `skip_output_layer`), and (c) only help a path blocked by the
fault above. The useful BioReason follow-ups are unrelated (replication/eval-harness gaps).

## Process notes (what went wrong, for next time)

- **Overclaimed before validating end-to-end.** Reported "mg1024 memory-feasible / validated"
  and pushed to main while the production run crashed at step 8. The memory win was real but
  conflated with end-to-end success. Corrected the report this session.
- **Chased deterministic mechanisms on a stochastic fault.** Should have repeated the *same*
  config early to test determinism before theorizing five mechanisms across three jobs.
- **`git commit -- <path>` swept pre-existing WIP** into commits (committed call sites without
  their library halves → two TypeErrors at HEAD; repaired in `b60fea56`). Stage hunks
  explicitly or commit from a clean tree.
- Aurora 2N infra was flaky (3 TCPStore rendezvous timeouts); 1-node runs dodged it.

## Cross-refs
- `docs/reports/colocate_memory_varlen_and_chunked_vocab_20260624.md` (the fixes, now corrected)
- `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md` (the blocker)
- memory: `project_linear_grpo_loss_chunked_vocab_20260624`,
  `project_colocate_mg1024_ccs_pagefault_20260624`,
  `project_colocate_logit_materialization_footprint_20260624`
