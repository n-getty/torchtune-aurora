# AGPT-2B GRPO — 8-node HSDP scale-up (distinct-prompt data parallelism)

**Date:** 2026-06-16
**Status:** VALIDATED end-to-end. 150/150 steps clean (rc=0), beats the 2N baseline on
both throughput and learning.

## What changed and why

The prior "2N production" AGPT-2B GRPO path was **1 single-node FSDP training replica + 1
vLLM node**, `batch_size=1` → exactly **one distinct prompt per optimizer step** (data-parallel
degree = 1). That is the unusual part: every production RL post-training stack feeds a *batch
of distinct prompts* to each step; GRPO's `G` samples handle within-prompt advantage
normalization but do not substitute for prompt diversity. The single-prompt regime gives a
high-variance policy gradient.

This work scales to the standard **disaggregated data-parallel topology**:

```
8 nodes (debug-scaling):
  Nodes 0-6 → 7 training replicas   dp_replicate=7 × dp_shard=12 (world=84)
              dense Llama3 FSDP1 HYBRID_SHARD within node; grads all-reduced
              across the 7 replicas (native HYBRID_SHARD inter-node all_reduce)
  Node 7    → 1 dedicated vLLM pool  12 HTTP servers, shared by all 7 replicas

Per optimizer step: batch_size(2) × dp_replicate(7) = 14 DISTINCT prompts (vs 1 on 2N)
Each replica's shard-leader POSTs its own prompt slice to the shared pool and
broadcasts completions node-locally over the gloo dp_shard PG.
Weight sync: only global rank 0 gathers the full model + broadcasts to the 12-server
pool each step (Llama Q/K un-permute applied).
```

## Headline result (apples-to-apples A/B; both wsync ON, G=8, on-policy ratios=1.0)

| Metric | 2N (job 8544461) | 8N HSDP (job 8545190) | Δ |
|---|---|---|---|
| Distinct prompts / step | 1 | **14** | 14× |
| Per-step wall | 12.25 s | 16.4 s | +34% |
| **Distinct-prompt throughput** | 0.082 /s | **0.85 /s** | **~10×** |
| Late-window mean reward (steps 100-150) | ~0.137 | **0.172** | +26% |
| Late-window mean success (steps 100-150) | ~0.044 | **0.067** | +53% |
| Train tiles / node | 11 | 12 | +1 |
| wsync_gather (rank 0) | 1.9 s | 0.6 s | faster |

8N processes ~10× the distinct prompts per second AND converges higher at equal steps — the
lower-variance gradient from 14 distinct prompts/step is the intended mechanism. Both runs
single-seed; the success delta (+53%) sits above the historical 4-7% SFT-variance band but a
2-seed replica would firm it up.

## Per-step timing decomposition (the +34%)

| Phase | 2N | 8N | note |
|---|---|---|---|
| gen (vLLM) | ~1.2 s | ~3 s | 7 replicas × G8 = 112 seqs/step share one vLLM node (vs 16) |
| grpo (ref+bwd) | ~7.0 s | ~11.0 s | fbs=4 → 4 backward chunks (2N fbs=8 → 2) ×2 FSDP collective rounds + cross-node gloo backward collectives |
| wsync_gather | 1.9 s | 0.6 s | rank-0 gather cheaper; broadcast overlaps |
| **total** | **12.3 s** | **16.4 s** | |

The slowdown is NOT the cross-replica grad all-reduce per se — it is (a) the extra chunked-
backward collectives from fbs=4 and (b) shared-vLLM-pool contention.

**fbs=8 to recover the grpo cost was tested (job 8545370) and FAILS:** it re-triggers the
reserved-pool fragmentation (banned:1 at step 12; torch_resv climbed to 47–57 GiB spread vs
fbs=4's flat ~30). fbs=8 doubles the per-chunk activation tensors, and those larger
variable-length chunk buffers fragment too — so fbs is load-bearing for memory, not just an
incidental setting. **The 34% per-step slowdown is therefore the cost of the topology at the
memory-safe envelope (G=8/bs=2/fbs=4); it is not recoverable via fbs.** The envelope is tight:
any of {bs=4, G=16, fbs=8} pushes the unlucky replica's reserved pool over the 64 GiB tile.
Remaining throughput levers are a 2nd vLLM node (gen contention) or `expandable_segments`
(blocked: oneCCL-incompatible on XPU).

## Validated production envelope

- Topology: `select=8` → 7 train replicas (`data_parallel_replicate_dim=7`, dp_shard=12) + 1 vLLM node
- `grpo_samples=8`, `batch_size=2`, `forward_batch_size=4`, `ref_forward_batch_size=4`,
  `max_generated_tokens=512`, `max_seq_len=1024` (this exact envelope is validated; fbs=8,
  G=16, and bs=4 each re-trigger banned:1 fragmentation)
- `vllm_weight_sync=true`, `method=xccl`, `interval=1`; `WSYNC_CROSS_METHOD=gloo`
- `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8` (no max_split_size)
- `TORCHTUNE_USE_CHUNKED_LOSS=0` (chunked backward)
- Memory: torch_resv flat ~30 GiB of 64 (huge headroom); no banned:1.

**Memory note:** per-rank buffer volume = `batch_size × grpo_samples` = sequences/rank.
G=8/bs=2 (16 seqs/rank) is the validated point. G=16/bs=2 (32 seqs/rank) re-triggers
per-replica data-dependent reserved-pool fragmentation → banned:1 at step ~12 (job 8545177).
Do not raise the product above 16 without re-validating memory.

## Artifacts

- Config: `recipes/configs/dev/production/auroragpt_2b_grpo_7n_gsm8k_hsdp_xpu.yaml`
- Launcher: `experiments/auroragpt_2b_bakeoff/run_agpt2b_7n_hsdp_8node.sh` + wrapper
  `_agpt2b_7n_train_rank_wrapper.sh` (qsub-only — mpiexec --pmi=pmix needs the PBS process
  tree; node-count-flexible: `select=N` → N-1 replicas + 1 vLLM)
- Recipe changes (all `dp_replicate==1`-guarded → 2N path byte-identical): `_setup_data`
  HSDP-aware sampler; `_generate_with_vllm`/`_broadcast_query_responses` per-replica
  shard-leader gen + gloo node-local broadcast; vLLM clients on every shard-leader;
  `_setup_model_fsdp1_hsdp` sets `_fsdp2_param_groups_meta`; `distributed.py`
  `enable_fsdp1_hsdp_inter_node_gloo` (FSDP1 HYBRID_SHARD inter-node all_reduce gloo reroute)
- Tests: `tests/torchtune/dev/rl/test_hsdp_server_sampler_contract.py`,
  `test_fsdp1_hsdp_inter_node_gloo_patch.py`
- Memory: `memory/project_agpt2b_7n_hsdp_launcher.md`

## Bring-up notes (debugging history)

Six launcher/recipe bugs during bring-up (each peeled one layer deeper): bare-name hostfile →
`:N` suffix / `--no-vni` → SSH-vs-qsub execution model → mom-node module load → wrapper
WORLD_SIZE resolution → dense-HSDP `_fsdp2_param_groups_meta`. Then a banned:1 endurance bug
mis-diagnosed twice (inter-node all_reduce; OFI-MR) before the mem-probe showed it was PyTorch
reserved-pool fragmentation from oversized per-rank buffers — fixed by the G×bs=16 seqs/rank
envelope. Lesson: with banned:1, read the mem-probe (torch_resv vs torch_alloc vs external)
before hypothesizing a transport/leak fix.

## Follow-ups

- 2nd seed to tighten the learning delta (single-seed today).
- Longer run: 8N reward was still climbing at step 150 (0.469); AGPT-2B keeps improving past
  150 (cf. mathmix plateau-break work).
- 2nd vLLM node would halve gen contention if generation becomes the bottleneck.
- Per-replica async generation + weight-sync overlap (not yet implemented).
