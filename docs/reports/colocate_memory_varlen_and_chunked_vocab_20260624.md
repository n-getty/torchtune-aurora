# Colocate training-memory fixes: varlen-cache leak + chunked-vocab LinearGRPOLoss (2026-06-24)

Two independent memory fixes for LoRA-GRPO `vllm_mode=colocate` on Aurora/XPU, plus a
correction to the "why does Aurora OOM where A100 doesn't" analysis. Target envelope:
Qwen3-4B LoRA, `max_generated_tokens=1024`, 64 GiB tile (the BioReason/paper goal).

## TL;DR

| Lever | Status | Effect |
|-------|--------|--------|
| `_varlen_out_cache` unbounded retention | **FIXED** (`attention_utils.py`) | killed a ~0.44 GiB/step LIVE-memory leak in the ref forward |
| Chunked-vocab `LinearGRPOLoss` (training fwd) | **memory win real; mg1024 NOT validated** | training backward peak 57.75 → 41 GiB (−16.7); but the mg1024 colocate run does NOT survive — see CORRECTION below |
| Explicit-mask attention S² | **already handled** | `TORCHTUNE_MASKFREE_CAUSAL=1` engages the fused bf16 FlashAttentionXPU kernel at bs=1 |
| Base / BioReason ports | **fail-fast guarded** | not portable as-is (FSDP FULL_SHARD / HF backbone); guarded against silent mis-wiring |

> ## CORRECTION (2026-06-24, post-diagnosis) — mg1024 colocate does NOT work
> An earlier version of this report claimed LinearGRPOLoss was "HW-validated at mg1024" and
> that "mg1024 now fits with ~24 GiB headroom; the residual is a separate L0 co-tenancy fault."
> **Both claims were wrong.** The mg1024 LoRA colocate run crashes (~step 7-8) with a
> `CCS NotPresent / PDE banned:1` GPU page fault. Follow-up diagnosis (sweep job 8559141 +
> A/B job 8559162) established the fault is **NON-DETERMINISTIC** (same `max_gen=512` config
> crashed once and survived 9 steps back-to-back) — NOT memory, NOT the co-tenancy fault
> (that's the two-process Ray TP=8 path), NOT our loss code. It is a probabilistic
> vLLM-on-XPU KV-paging fault whose probability rises with generation volume. See
> `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md`.
>
> **Honest status:** the two memory fixes below are real and on main (opt-in, default path
> unchanged). The `max_gen=256` colocate envelope is the low-risk practical ceiling.
> `max_gen ≥ 384` colocate is flaky and `max_gen=1024` is **blocked** by the driver-level
> fault — independent of these fixes. LinearGRPOLoss is a validated *memory optimization*, not
> a validated mg1024 production path.

Net for LoRA colocate: the two memory fixes remove real memory walls, but mg1024 colocate
remains blocked by the non-deterministic generation page fault (a vLLM/XPU driver issue, not
a memory-capacity problem).

## 1. The `_varlen_out_cache` leak (root cause of the "0.44 GiB/step creep")

**Symptom:** no-FSDP colocate (`TORCHTUNE_COLOCATE_NO_FSDP=1`) made `empty_cache` reclaim
cleanly (no more FSDP UR-handle leak), but a ~0.44 GiB/step residual creep remained on the
post-reclaim floor — linear, unbounded, OOM at ~step 51.

**Misdiagnoses ruled out by measurement** (each refuted the prior guess):
empty_cache/L0 residue (A/B: creep byte-identical at 5 vs 1 `empty_cache`/step) → load_weights
(ACTIVE per-phase: sync = +0.000) → vLLM generate (ACTIVE per-subcall: generate = +0.000) →
**the reference forward** (ref_fwd ACTIVE = +0.44/step).

**Root cause:** the IPEX-varlen no-grad output-buffer cache in
`torchtune/modules/attention_utils.py` (`_varlen_out_cache` / `_varlen_alibi_cache` /
`_varlen_seqlens_cache`), added 2026-04-30 for a *fixed*-seqlen microbench ("0 allocator
delta per call" by reusing the output buffer). Keyed by `(b, h, s, d, dtype, device)`. On
**variable-seqlen RL** (GSM8K rollouts: prompt+completion length differs nearly every step),
`s` changes per step → a fresh `~14.7 MiB` buffer (`[b·s, h, d]` bf16) × n_layers cached
**forever**. It is a module-level dict so gc sees it, but `empty_cache` can't free it (live).
The growing tensor exactly matched the run-8 census `[S, 32, 128]×36`.

**Fix:** `OrderedDict` + FIFO eviction (`_varlen_cache_evict`), cap = `TORCHTUNE_VARLEN_CACHE_MAX`
(default 8). All three caches evicted jointly by shared key. Within-step reuse (all layers
share one key) and consecutive-same-shape reuse are preserved → the varlen speedup is kept;
only stale `(b, s)` shapes are evicted.

**Validation (A/B, job 8558656, mg256, same nodes):** varlen=1 → ref_fwd ACTIVE +0.44/step;
varlen=0 → +0.000, ALLOC floor dead-flat 22 steps. Fix run (8558690): varlen ON + bounded →
ALLOC plateaus ~34.5 GiB, GREEN, step time unchanged (~8.1s). **Affects ALL variable-seqlen
runs with `TORCHTUNE_USE_IPEX_VARLEN=1` (server/dedicated GRPO too), not just colocate.**

Tests: `tests/torchtune/dev/rl/test_varlen_buffer_cache.py` (bounded ≤ cap across cap+20
distinct shapes; same-shape stays 1 entry; original reuse invariants intact).

## 2. Chunked-vocab `LinearGRPOLoss` (training-forward logit memory)

**Why:** after the leak fix, the dominant remaining mg1024 cost is the FP32 full-vocab logit
materialization in the **training** forward: `transformer.py` `self.output(h).float()` →
`[B, S, vocab]` FP32 (~2.7 GiB/seq at S~1900, Qwen3-4B vocab=151936), then `log_softmax` a 2nd
FP32 copy. With `TORCHTUNE_USE_CHUNKED_LOSS=1` (per-chunk fwd+bwd at fbs=1) the backward graph
is bounded, but the per-sequence vocab tensor still dominates and scales with the longest rank.

**Fix:** wire the existing-but-unused `torchtune/dev/rl/linear_grpo_loss.py` `LinearGRPOLoss`.
With `set_model_output(model)` the model's `skip_output_layer=True` makes the training forward
return **hidden** states `[B, S, 2560]` (59× smaller); the loss applies the vocab projection
**per sequence-chunk** and reduces each to logprobs before freeing it — the full `[B,S,vocab]`
tensor is never held.

**Opt-in, default unchanged:** detected via `isinstance(loss, RLLoss) and
hasattr(loss, "set_model_output")`. The shipped `GRPOLoss`/`GRPOSimpleLoss` lack
`set_model_output` → the existing full-logit path runs byte-for-byte unchanged. Select via the
loss component (`loss._component_=torchtune.dev.rl.linear_grpo_loss.LinearGRPOLoss`).

**Behavioral side-effects (all bit-equivalent on the validated envelope, else fail-fast):**
- *Loss math:* LinearGRPOLoss is the GRPOSimple formulation (no IS-ratio clip). At
  `ppo_epochs==1` + on-policy (`always_compute_rollout_logprobs==False`), `pi_old==pi.detach()`
  so GRPOLoss's clip is inert too → bit-identical gradients (CPU test). **Fail-fast** otherwise.
- *Temperature:* threaded into the loss CE (`log_softmax(logits/T)`), matched to the recipe
  temperature — unblocks the T=0.7/0.8 configs.
- *KL hardening:* `clamp[-10,10]`+`nan_to_num` (mirrors `loss.py`), survives long-gen -inf
  logprobs.
- *FSDP / compile:* **fail-fast** unless `TORCHTUNE_COLOCATE_NO_FSDP=1` (projection runs
  outside `model.forward`; FULL_SHARD reshards the weight → wrong numerics) and `compile=False`
  (skip_output_layer toggle changes the forward return type → breaks compile guards).

**Throughput:** neutral. grpo (train fwd+bwd) component 3.70s vs 3.91s (LinearGRPOLoss slightly
faster, within noise); total step 29.1 vs 29.7s. Chunking the projection into 8 GEMMs adds no
measurable overhead (same FLOPs, smaller working set, avoids the giant FP32 alloc/free).

**HW run (job 8558962, mg1024, LoRA colocate) — memory measurement valid, run did NOT survive:**
- training backward peak **57.75 → 41.05 GiB** (−16.7) — the memory win is real and measured.
- the run nonetheless **crashed ~step 7-8 during generation** (`banned:1` page fault, 24 GiB
  free — NOT OOM, NOT the changed backward). This was initially mislabeled an "L0 co-tenancy"
  fault; that was WRONG (co-tenancy is the two-process Ray TP=8 path). Follow-up diagnosis
  (jobs 8559141, 8559162) established it is the **non-deterministic vLLM-XPU generation page
  fault** documented in `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md` — same
  fault hits the GRPOLoss baseline and predates these changes. The "per-rank seqlen variance
  absorbed" observation was also an artifact of fitting a stochastic fault; disregard it.
- **Conclusion: LinearGRPOLoss reduces colocate training memory as designed, but does NOT make
  mg1024 colocate survive — that path is blocked by the driver-level fault, not memory.**

Tests: `test_linear_grpo_loss_equivalence.py` (forward+gradient bit-equivalence vs
GRPOSimpleLoss for chunks {1,2,4,8} incl uneven split; temperature {0.7,0.8,1.3};
memory-shape; KL finiteness) and `test_linear_grpo_loss_recipe_guards.py` (fail-fasts present).

## 3. Why Aurora OOM'd where A100/TRL (40 GiB) didn't — four mechanisms, not "fit"

It was never a 40-vs-64 GiB fit problem. The colocate OOM was the sum of four separable causes:
(a) the varlen-cache leak [fixed]; (b) single-vs-chunked backward [`USE_CHUNKED_LOSS=1`];
(c) FP32 full-vocab logits [LinearGRPOLoss]; (d) per-rank seqlen variance [absorbed by (c)].
**Correction:** XPU *does* have a fused autograd FlashAttentionXPU kernel for bf16
(`mask=None + is_causal=True`); the explicit `[b,1,S,S]` mask forces the O(S²) math path, and
`TORCHTUNE_MASKFREE_CAUSAL=1` (engages at bs=1, no prompt padding) already routes the training
forward onto the fused kernel — verified job 8558815 (no "prompt padding" bail).

## 4. Port status (base + BioReason)

Both are **fail-fast guarded**, not ported — the projection-outside-forward approach is unsafe
there:
- **base full-FT** (`grpo_full_finetune_distributed_xpu.py`): FSDP FULL_SHARD + a **trained**
  tied output projection → resharded weight gives wrong numerics + broken grads. No no-FSDP path.
- **BioReason** (`grpo_bioreason_distributed_xpu.py`): HF `AutoModelForCausalLM` backbone has no
  torchtune `skip_output_layer` hidden path; also FSDP FULL_SHARD.

A FULL_SHARD-safe (summon-based) projection is future work (Phase 3). The LoRA colocate recipe
(no-FSDP, frozen tied output) is the supported home for the chunked-vocab path.

## New env flags

| Flag | Default | Effect |
|------|---------|--------|
| `TORCHTUNE_VARLEN_CACHE_MAX` | 8 | FIFO cap on the IPEX-varlen output-buffer caches (bounds the variable-seqlen retention) |
| `TORCHTUNE_LINEAR_LOSS_ALLOW_FSDP` | 0 | override the LinearGRPOLoss no-FSDP requirement (only safe under SHARD_GRAD_OP; untested) |

## Memory references
`project_nofsdp_colocate_creep_diagnosis_20260623`,
`project_colocate_logit_materialization_footprint_20260624`,
`project_linear_grpo_loss_chunked_vocab_20260624`.
