# Chunked-vocab `LinearGRPOLoss` in the base FSDP2 GRPO recipe (2026-06-25)

Makes the memory-efficient chunked-vocab `LinearGRPOLoss` correct and opt-in in the
**base** distributed GRPO recipe (`recipes/dev/grpo_full_finetune_distributed_xpu.py`),
where it previously fail-fasted. Until now it was wired only in the LoRA *no-FSDP*
colocate recipe; the `[B,S,vocab]` FP32 logit cost it removes is present in **all** GRPO
recipes, and the base FSDP2 server/dedicated path is the higher-value home for it.

Phase 1 scope: FSDP2 FULL_SHARD, non-EP, non-HSDP, non-packing, no-compile,
`ppo_epochs==1` + on-policy, tied-embedding models (qwen2_5_3b / qwen3_4b). Everything
outside that scope fail-fasts. The default `GRPOSimpleLoss`/`GRPOLoss` path is
byte-for-byte unchanged (the loss is only wired when the config selects a loss exposing
`set_model_output`).

## TL;DR

| Item | Result |
|------|--------|
| Correctness (fp32) | **bit-exact** to the full-logit reference (grad max_abs_diff = 0.0, 2-rank probe) |
| Correctness (bf16, HW) | grad_norm parity 0.3% (ref 0.4807 vs lin 0.4823, identical rollout); pure path-noise |
| Memory (2N, rank-0, step-1) | peak_active **−0.59 GiB (8.5%)**, peak_reserved **−3.42 GiB (27.4%)** at a small envelope; scales with `num_seqs×S×vocab` |
| Throughput | **no step-time change** — same FLOPs; the win is memory headroom, not speed |
| Tied-embedding mechanism | weight stays resident in the **root** FSDP2 unit; do **NOT** make `tok_embeddings` its own `custom_sharded_layers` unit |
| Run health | both A/B legs **GREEN** (`scripts/check_run_health.sh`), 1N + 2N, rc=0 |

## What the optimization does

With `model.skip_output_layer=True` the transformer returns post-norm **hidden** states
`[B,S,emb]` instead of logits. `LinearGRPOLoss` then applies the vocab projection
**per sequence-chunk** inside `chunked_grpo_loss`, reducing each chunk to logprobs and
freeing it — so the full `[B,S,vocab]` FP32 logit tensor (~2.7 GiB/seq for Qwen3-4B at
S~1900, vocab=151936) is **never materialized**. `skip_output_layer` is toggled True
only around the training forward in `grpo_step`; generation and the reference forward
keep returning logits (the in-forward tied unembed), unchanged.

## The tied-embedding correctness problem (the make-or-break)

Targets are tied: `output = TiedLinear(tok_embeddings)` (qwen2/qwen3 with
`tie_word_embeddings=True`). `TiedLinear` is not an `nn.Module`; the projection weight
physically lives in `model.tok_embeddings`. The loss projects **outside** `model.forward`,
so the weight must be full (not a shard) at projection time.

**The plan's original assumption was wrong and harmful.** It assumed tied models needed
`custom_sharded_layers: ['tok_embeddings']` so the loss could `unshard()` that unit. HW
testing (job 8559361) showed that making `tok_embeddings` its own `fully_shard` unit
**reshards it mid-forward**, so the *in-forward* tied unembed used by generation/ref
(skip_output_layer=False) hits a sharded DTensor weight:

```
RuntimeError: aten.mm.default got mixed torch.Tensor and DTensor ...   # in generate()
```

**Root cause, established by two throwaway probes (not by guessing):**

- `probe_tied_resident.py` — under **default** torchtune sharding (no
  `custom_sharded_layers`) `tok_embeddings` is **not** its own FSDP unit; it stays in the
  **root** FSDP2 unit. The root uses AllGather prefetch (`reshard_after_forward=None`), so
  its own params stay **resident** through the forward → `tok_embeddings.weight FULL=True`
  post-forward → `F.linear(hidden, weight)` works with no manual unshard.
- `probe_tied_grad.py` — the chunked-vocab projection + backward grad on the tied
  `tok_embeddings.weight` is **bit-exact** (max_abs_diff = 0.0) to a single-process
  full-weight reference. Tied-grad (input-embedding use + unembed use → one parameter) is
  accumulated correctly by FSDP2 autograd. This was the design's #1 risk; cleared.

**Correct design (implemented):**

- `set_model_output` (tied): `linear_projection = lambda x: F.linear(x, tok_embeddings.weight)`
  (closure reads the *current* weight) and capture `self._fsdp_root = model` (the root
  FSDP module). `forward()` calls `self._fsdp_root.unshard()` once before the chunk loop —
  a **no-op** under default prefetch (root already resident), and the correctness fix only
  if the root ever reshards (`disable_prefetch=True`). It does **not** unshard
  `tok_embeddings` as a separate unit.
- `set_model_output` (untied): capture `model.output` directly (its own `fully_shard`
  unit; the post-forward call fires the all-gather hook). Requires `'output'` in
  `custom_sharded_layers`.
- Recipe fence: for tied models, **forbid** `'tok_embeddings'` in `custom_sharded_layers`
  (fail-fast); for untied, **require** `'output'`.

## Why fp32 is bit-exact but bf16 differs ~1%

The reference path computes logprobs as `log_softmax(logits)` over the full vocab; the
chunked path computes `-cross_entropy` per chunk. These are mathematically identical
(fp32 probe: rel = 0.000%) but use different arithmetic-reduction orders, so in **bf16**
they differ by rounding noise that grows with vocab/seqlen. `probe_bf16_pathdiff.py` (CPU)
measures rel ≈ 0.22% on toy dims; the real model shows ~0.3–1.6% grad_norm delta at
step 1. This is **expected numerical noise, not a bug** — pinned by
`test_bf16_path_noise_is_bounded` (fp32 < 1e-5, bf16 < 5e-2). The plan's "~1e-3" target
was unachievable for a bf16 full-vs-chunked-vocab comparison; the correct criterion is
fp32 bit-exact + bf16 bounded + matching dynamics, all of which hold.

## HW validation (Aurora XPU, qwen2_5_3b, GSM8K, native generation)

**1-node** (job 8559361, 2-tile + 8-tile FSDP2, dp_replicate=1):
- ref (`GRPOSimpleLoss`) and lin (`LinearGRPOLoss`) both 5/5 rc=0; CHUNKED + SINGLE
  backward paths both exercised; 8-tile (real sharding) grad_norms healthy
  (1.56/1.12/1.35/2.28). SINGLE_BACKWARD lin → `check_run_health` **GREEN**.

**2-node cross-node** (job 8559527, mpiexec --pmi=pmix, WORLD=24, dp_replicate=1,
SINGLE_BACKWARD, G=16 / max_gen=128 / batch_size=1, both legs back-to-back same nodes):

| metric (rank-0, step 1, identical rollout) | ref full-logit | lin chunked-vocab | saved |
|---|---|---|---|
| `peak_memory_active`   | 6.96 GiB  | 6.36 GiB | **0.59 GiB (8.5%)** |
| `peak_memory_reserved` | 12.49 GiB | 9.07 GiB | **3.42 GiB (27.4%)** |
| `grad_norm`            | 0.4807    | 0.4823   | 0.3% (bf16 noise) |
| fwd / bwd time (rank0) | 1.5s / 6.5s | 1.5s / 5.7s | ~same (node noise) |

Both legs **GREEN**, rc=0.

- The **reserved** delta is the real signal: the transient full `[num_seqs,S,vocab]` FP32
  logit pool the standard path reserves is never allocated by the chunked path. The
  saving scales with `num_seqs × S × vocab`; this run used a deliberately small envelope
  (`max_gen=128`) to fit cross-node generation speed, so at the production envelope
  (S~1900) the saving is multiple GiB/rank.
- **Throughput is unchanged** — same arithmetic, just sliced per chunk. This is a memory
  optimization (enables larger batch/seq, avoids OOM), not a speed optimization. Stated
  honestly so it is not mistaken for a throughput lever.

### Measurement / HPC notes
- 2N `mpiexec --pmi=pmix` must run as a **PBS batch job** — PALS RPC launch fails from an
  SSH'd compute node. Hostfile must be plain FQDNs from `$PBS_NODEFILE`.
- Native generation is ~1.1 s/token cross-node (per-token FSDP allgather), so keep
  `max_gen` small for memory/correctness A/Bs; it does not affect the logit-memory result.
- Per-step peak is from the rank-0 DiskLogger (`peak_memory_active/reserved`, reset each
  step); the earlier `max_gen=64 / fbs=1` run showed only ~0.06 GiB because the logit was
  ~38 MB — the saving only appears once the materialized logit is large
  (SINGLE_BACKWARD + larger `grpo_samples`).

## Files

- `torchtune/dev/rl/linear_grpo_loss.py` — tied-aware `set_model_output` + root unshard
- `recipes/dev/grpo_full_finetune_distributed_xpu.py` — `_wire_linear_grpo_loss()` (scope
  fences) + linear path in SINGLE_BACKWARD & CHUNKED_BACKWARD branches
- `recipes/configs/dev/baseline/qwen3B_grpo_xpu_linearloss.yaml` — opt-in config (no
  `custom_sharded_layers`)
- Tests: `tests/torchtune/dev/rl/test_linear_grpo_loss_equivalence.py`,
  `test_linear_grpo_loss_recipe_guards.py`
- Validation harness (gitignored under `experiments/baselines/`):
  `probe_tied_resident.py`, `probe_tied_grad.py`, `probe_bf16_pathdiff.py`,
  `run_linearloss_2n.sh`, `pbs_linearloss_2n_ab.sh`, `extract_2n_mem.py`

## Out of scope (future)

Reference-forward memory (still full-logit); EP; HSDP / FSDP1; packing; untied models
(SFT-style `'output'` unit path is wired but unverified on HW); BioReason (HF backbone has
no `skip_output_layer`). The LoRA-colocate mg1024 path is blocked by a separate
non-deterministic vLLM-XPU page fault (`docs/bugs/xpu_colocate_generation_pde_nondeterministic.md`).
