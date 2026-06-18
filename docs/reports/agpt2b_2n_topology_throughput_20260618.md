# Best 2N AuroraGPT-2B GRPO setup — vLLM-placement topology study

**Date:** 2026-06-18 · **Task:** sum_digits · **Hardware:** 2 Aurora nodes (24 PVC tiles)
· **Envelope:** G=16, batch_size=1/rank, max_gen=64, temp=0.7, 50 steps.

## Question

On exactly 2 Aurora nodes, what is the best torchtune GRPO setup for AuroraGPT-2B, and
where should vLLM generation live relative to training? Four topologies, each at its
natural `data_parallel_replicate_dim`, compared on per-step throughput (steady-state
`TIMING_DETAIL`) and convergence (50-step sum_digits success).

## Throughput (common G=16 envelope, steady-state s/step)

| topology | dp_rep | train tiles | vLLM | **total** | gen | grpo | wsync | health |
|----------|:------:|:-----------:|------|:---------:|:---:|:----:|:-----:|--------|
| colocate-flat   | 1 | 24 | in-process/rank | 7.8 | 1.4 | 5.1 | ~2.0 | GREEN |
| **colocate-HSDP** | **2** | **24** | in-process/rank | **5.5** | 0.7 | 4.1 | 0.7 | GREEN |
| 11+1 per-node   | 2 | 22 | 1 server/node | 5.9 | 0.6 | 4.1 | 0.6 | GREEN |
| dedicated-node  | 1 | 12 | 12 servers (node 1) | 7.6 | 0.7 | 0.5 | 5.9† | gloo-wsync‡ |

† dedicated-node is dominated by `wsync_prev_wait` (cross-node full-model push to the
12-server pool), not training compute (`grpo`=0.5 s on only 12 ranks). ‡ The health gate
flags the cross-node gloo `reduce_scatter` (timing caveat only; accuracy unaffected).

**The dominant 2N lever is `dp_replicate=2`, not vLLM placement.** colocate-flat →
colocate-HSDP (same 24 tiles, same in-process vLLM, only `dp_replicate` 1→2) cuts step
time **7.8 → 5.5 s (−29%)** by keeping the per-layer FSDP shard collective **intra-node**
(12-way XCCL) instead of spanning both nodes (24-way cross-node AllGather/ReduceScatter).
The two fastest topologies are both `dp_replicate=2`; the two slowest both `=1`.

## Convergence (each topology at its correct envelope, sum_digits 50 steps)

| topology | envelope | kl median | kl max | NaN | best success |
|----------|----------|:---------:|:------:|:---:|:------------:|
| colocate-flat   | kl=0, lr=1e-5 | 0.17 | 0.28 | 0 | 0.80 |
| colocate-HSDP   | kl=0, lr=1e-5 | 0.13 | 0.31 | 0 | **1.00** |
| 11+1 per-node   | kl=0.02, lr=5e-6 | 0.10 | 1.63 | 0 | 0.44 |
| dedicated-node  | kl=0.02, lr=5e-6 | 0.063 | 2.10 | 0 | 0.50 |

**Server topologies require the stable envelope.** colocate tolerates kl_coeff=0/lr=1e-5;
server-mode generation does **not** — at kl=0/lr=1e-5 the policy diverges (kl → 1e7, NaN
by ~step 25). This is a documented AGPT-2B server characteristic (the production config
`auroragpt_2b_grpo_2n_gsm8k_xpu_real.yaml` halves lr and adds kl_coeff=0.02 "because
baseline 0.0 NaN'd by step 40"). With the stable envelope (kl_coeff=0.02, lr=5e-6,
`always_compute_rollout_logprobs=true`), both server topologies train cleanly: bounded kl,
0 NaN, reward climbing to 0.84–0.88.

Server convergence (best success 0.44–0.50) is lower than colocate-HSDP (1.00) because the
server topologies have fewer training ranks (22 / 12 vs 24) and fewer distinct
prompts/step (shard-leader/rank-0 generation at bs=1 vs colocate's per-rank generation).
That is an expected topology property at this envelope, not a defect.

## Recommendation

**Best 2N AGPT-2B GRPO setup: flat colocate with `data_parallel_replicate_dim=2`** — 24
training tiles, in-process vLLM, two intra-node 12-way FSDP shards. It is the fastest
(5.5 s/step), the most stable (best success 1.00, kl bounded), the simplest (no separate
vLLM node or launcher), and memory-comfortable (~29 / 64 GiB per tile). The server
topologies (11+1, dedicated node) are viable and now stable, but slower and/or
lower-converging on a memory-comfortable 2B; they earn their keep only when the model is
too large to colocate or generation dominates the step (long max_gen / large G), neither
of which holds for AGPT-2B / sum_digits.

## Code landed during this study

- **Bug fix (regression):** commit `f60efefc` (2026-06-17 wsync refactor) dropped the
  broadcast for the FSDP1 + `vllm_weight_sync_method=xccl` path — `_xccl_gather_fsdp1`
  gathered weights then returned without sending them, so vLLM never received updates →
  frozen-generator drift → ∇logp blowup → NaN. **Without this fix, every server topology
  NaNs.** Restored in `torchtune/dev/rl/weight_sync.py` (`_xccl_gather_fsdp1`): greedy CPU
  batching matching the receiver's `batch_max_numel` contract + `_deferred_broadcast_args`.
  Regression guard: `tests/torchtune/dev/rl/test_xccl_fsdp1_broadcast_present.py`.
- **FSDP1 colocate weight sync** (earlier this session): `_sync_colocated_weights` now
  supports FSDP1 HYBRID_SHARD (`summon_full_params` + `_fsdp_wrapped_module` prefix strip),
  required for colocate-HSDP (`dp_replicate>1`).
- **Launcher hardening:** `2N_11plus1_sweep.sh` now retries vLLM startup on the Aurora
  EngineCore flake (health + PID liveness, relaunch-once per node).
- Full rl test suite: **466 passed**.

## Artifacts
- Configs: `recipes/configs/dev/production/auroragpt_2b_grpo_2n_sumdigits_{colocate,colocate_hsdp,server}_xpu.yaml`
- Launchers: `experiments/auroragpt_2b_bakeoff/{2N_sweep.sh, 2N_server_sweep.sh, 2N_11plus1_sweep.sh}`
- Raw results: `experiments/auroragpt_2b_bakeoff/logs/RESULTS_topology_FINAL.txt` + per-cell run dirs.

## Method notes (for reproducers)
- Throughput is measured at a common G=16 envelope; convergence at each topology's stable
  envelope. The two are reported separately because server mode needs kl_coeff>0/lower lr
  to converge while colocate does not — comparing convergence at a single shared envelope
  would unfairly penalize whichever mode that envelope destabilizes.
- HPC: export `PBS_JOBID` *inside* the ssh command (it does not propagate as an env prefix
  → PALS "Job not found"). If a login-node SSH hangs at the banner, `unset SSH_AUTH_SOCK`
  (a wedged agent socket stalls the handshake).
