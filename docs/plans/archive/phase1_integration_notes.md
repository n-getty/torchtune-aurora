# Phase 1 integration notes — async generation, server mode

Context: Phase 0 (logprobs always populated, gated by config) has landed.
Phase 1 is "deeper deferred" — overlap the next batch's vLLM HTTP generate
call with the current batch's training. Server mode only.

## What can actually overlap

`generate_trajectory()` does (in order):
1. `_generate_with_vllm` — HTTP POST to vLLM server, broadcast result to
   all ranks. Only step that's parallelizable with training.
2. Reference policy forward — XPU collective (`_ref_model(query_responses)`).
3. Phase-0 policy forward — XPU collective when
   `_compute_rollout_logprobs_required` is true.
4. Reward computation (CPU/decoded strings) + advantage normalization.

For Phase 1, only step 1 can run in a producer thread. Steps 2-4 must run
synchronously on the consumer side because they need every training rank
to participate in collectives.

This means Phase 1's overlap window is `vllm_gen_time` only — not the full
`generate_trajectory` cost. Worth measuring on the 3B sync baseline first
to see how much of step time the vLLM piece actually consumes. Earlier
log entries report `vllm=` separately in `GENTIMING`; those numbers tell
us the upper bound on Phase-1 speedup.

## Shape of the producer

Producer (rank 0 only):
- Owns its own dataloader iterator (separate from train loop's, sampler
  rank=0 num_replicas=1) so it can pull batches independently.
- For each batch: extract `tokens`, expand to `[B*G, P]` (mirrors the
  in-line code at L5260-5261), call `_generate_with_vllm(...)` which
  returns the `query_responses` tensor on rank 0 only.
- Push `(batch, query_responses_cpu)` onto a `queue.Queue(maxsize=1)`.

Consumer (all training ranks):
- At top of step: rank 0 pops from queue. `batch` and the cached
  `query_responses` are then broadcast as objects + tensors to all ranks.
- Continue with ref_fwd, policy_fwd (Phase 0 path), reward, advantages.
- Trigger weight sync (existing path) after optimizer step.

## Concerns

1. **Dataloader sharing**: rank 0 producer needs a *separate* dataloader
   that's deterministic vs the training-rank-0 dataloader so the same
   batches reach the same group baselines. Easiest is `torch.Generator`
   with the same seed and a single-shard `DistributedSampler(num_replicas=1)`.
   But: the existing recipe shards the dataloader across all training
   ranks. For a single-node 1.5B/3B prototype this doesn't really
   matter — for production multi-node we'd need to revisit.

2. **HTTP vLLM session reuse**: `_generate_with_vllm` uses
   `self._vllm_clients` which talks via `requests`. Verify thread-safety
   of `VLLMClient` — likely fine since each call constructs a fresh
   request, but worth a one-line check.

3. **Weight sync coordination**: after optimizer step, the weight sync
   (currently `_start_deferred_broadcast`) ships new weights to vLLM.
   The producer must NOT start the next batch until that sync completes,
   or the rollout will use stale-by-2 weights instead of stale-by-1.
   Bridge via the existing `_sync_done_event`: producer waits on it
   before issuing the next `generate()`.

4. **Error propagation**: producer thread crashes must surface to all
   ranks (not just rank 0) or training will deadlock waiting for a
   broadcast that never comes. Wrap producer call in try/except, set a
   shared `producer_dead` flag, broadcast it at consume time.

5. **GRPOSimpleLoss**: the proven 3B config uses GRPOSimpleLoss which
   doesn't use pi_old_logprobs. Phase 1 doesn't strictly require switching
   loss (k=1 with current=old means ratios collapse to 1, same as today),
   but Phase 2 (k>1) absolutely does. The async-proto config already
   switches to GRPOLoss.

## Files to modify (Phase 1)

- `recipes/dev/grpo_full_finetune_distributed_xpu.py`:
  - `_setup_vllm_server_mode` (~L1980): wire up `RolloutProducer` with
    `_generate_with_vllm` as `produce_fn`.
  - `train()` (~L6286): replace `trajectory = self.generate_trajectory_batched(...)`
    with `producer.get()` → broadcast → ref_fwd + policy_fwd path
    pulled out of the original `generate_trajectory`.
  - `cleanup()`: `producer.stop()`.
- `recipes/configs/dev/production/qwen3B_grpo_async_xpu.yaml`: flip
  `async_generation.enabled=true` for the Phase 1 run (already structured
  for it).
- `torchtune/dev/rl/async_rollout.py`: already exists.

## Verification

1. With `async_generation.enabled=true`: 5 steps complete, no deadlock,
   no crash; producer + consumer telemetry shows positive overlap.
2. Loss curve over 50 steps within run-to-run noise of Phase-0 baseline
   (logprobs path matches; ratios should be ≈ 1 since k=1 + same step
   weights).
3. Step time decreases by `~vllm_gen_time` (whatever GENTIMING reported
   on the sync baseline).
