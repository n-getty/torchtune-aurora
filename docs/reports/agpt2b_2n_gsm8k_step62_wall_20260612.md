# AGPT-2B GRPO 2-Node GSM8K: Step-62 Wall — Failure Tree, 7-Run Isolation, Open Hypotheses

**Date:** 2026-06-12
**Severity:** Blocks all AGPT-2B GRPO production runs > 61 steps on 2-node Aurora
**Affected:** AuroraGPT-2B GRPO, 2-node Aurora (11-tile FSDP train + 12-tile vLLM HTTP DP=12)
**Status:** Three correctness bugs root-caused and fixed; one resource leak deterministic and unfixable in the current stack. Working configuration good to step 61.

## Executive Summary

The first AGPT-2B GSM8K 2-node production run died at step 49 with a cascade of four issues stacked on top of each other. After tonight's session three of the four are root-caused and shipped as fixes; the fourth is a deterministic L0/XCCL collective stall at step 62 that persists across every configuration lever we have. **All seven independent variants crash at exactly step 62** with the same staircase pattern in the no-grad ref forward — ranks complete the forward at 3s / 23s / 33s / 43s / 53s / 63s, then SIGABRT.

This is in the same family as the canonical sig#2 leak (`docs/bugs/intel_xpu_resource_leak_bug_report.md`) but the per-step iteration count where it fires (~62 training steps × ~3 FSDP-allgather cycles per step ≈ 186 cycles) is its own data point — distinct from both the synthetic `empty_cache()` reproducer (FSDP2 iter ~70) and the 32B-FSDP2 `banned:1` crash at step 28-29 (`docs/reports/cxi_mr_step28_crash_investigation_20260428.md`).

**Bottom line**: training dynamics are now healthy — rewards trend up (0.06 → 0.31 by step 40), KL is bounded (0.02 → 0.4), memory is flat (~32 GiB). But the run cannot reach the `num_steps: 150` target without checkpoint/restart at step ~50.

## The cascading failure tree (initial 49-step crash)

The first run (`logs/gsm8k_2n_server_20260612_003819`) crashed at step 49 with `RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)`. Four distinct problems were entangled:

| # | Problem | Status |
|---|---------|--------|
| 1 | Llama Q/K un-permute had no log evidence of engaging at runtime | **Fixed** — added one-shot `ENGAGED` log on all 4 wsync paths |
| 2 | `TORCHTUNE_VARLEN_NOGRAD_BYPASS=1` drops ref-fwd mask → kl_loss = 6.7e7 at step 1 on un-converged ref | **Fixed** — default flipped to off on AGPT-2B path |
| 3 | `lr=1e-5` + `kl_coeff=0` → NaN by step 40 (KL unbounded gradient drift on GSM8K) | **Fixed** — stable YAML with lr=5e-6, kl_coeff=0.02, warmup=10 |
| 4 | Deterministic step-62 wall | **Open** — see below |

## Stability headline (post-fix)

After applying fixes 1-3, training is healthy through step 61:

| metric | step 1 | step 20 | step 40 | step 60 |
|--------|-------:|--------:|--------:|--------:|
| `rewards` | 0.062 | 0.188 | 0.156-0.31 | 0.0-0.156 |
| `kl_loss` | 0.0 | 0.08-1.9 | 0.02-2.2 | 0.13-0.91 |
| `loss` | ~0 | ~0.003 | 0.005-0.04 | ~0.003 |
| `peak_memory_active` | 28.2 GiB | 30.2 GiB | 32.3 GiB | 32.4 GiB |
| `peak_memory_reserved` | 44.7 GiB | 55.9 GiB | 61.3 GiB | 61.3 GiB |

`ratios=1.0` throughout — single-epoch on-policy GRPO collapses to REINFORCE-with-baseline by design (`_compute_rollout_logprobs_required=False` makes `pi_old = pi.detach()`). The KL term and reward signal drive the gradient via `pi_logprobs` autograd.

## The step-62 wall

### Crash signature

```
[default0]:Rank 0: SINGLE-BWD FAILED step=62 ... (in some runs)
[default0]:Signal 6 (SIGABRT) received by PID ...
```

More commonly the crash happens silently inside the no-grad ref forward at step 62, before any per-step log gets emitted. Pattern observed across all 7 runs:

```
Rank 10: GENTIMING vllm=4.1s policy_fwd=0.0s ref_fwd=3.0s    # most ranks
Rank 5:  GENTIMING vllm=4.1s policy_fwd=0.0s ref_fwd=23.0s   # +20s late
Rank 7:  GENTIMING vllm=4.1s policy_fwd=0.0s ref_fwd=33.0s   # +30s late
Rank 4:  GENTIMING vllm=4.2s policy_fwd=0.0s ref_fwd=43.0s   # +40s late
Rank 2:  GENTIMING vllm=4.2s policy_fwd=0.0s ref_fwd=53.0s   # +50s late
Rank 0:  GENTIMING vllm=4.1s policy_fwd=0.0s ref_fwd=63.1s   # +60s late, then SIGABRT
```

The 10-second-apart staircase across all 11 training ranks is the classic XCCL collective timeout sequence — one rank hangs in an FSDP allgather inside the ref forward; the other ranks block on the same collective and time out one by one at the operation's individual timeouts.

### What is NOT the cause (7-run isolation matrix)

| Job     | Variant                                                    | Last clean step |
|---------|------------------------------------------------------------|-----------------|
| 8538544 | unstable hparams (lr=1e-5, kl=0), FSDP2 FULL_SHARD         | 61 (NaN by 40)  |
| 8538722 | stable (lr=5e-6, kl=0.02), FSDP2, wsync int=1              | 61              |
| 8538788 | stable, FSDP2, wsync int=2 (halved wsync rate)             | 61              |
| 8539546 | stable, FSDP2, `WSYNC_CROSS_METHOD=gloo` (TCP over hsn0)   | 61              |
| 8539581 | stable, **FSDP1 SHARD_GRAD_OP** (ZeRO-2 flat)              | 61              |
| 8539607 | stable, FSDP1 ZeRO-2, **`vllm_weight_sync=false`**         | 61              |
| 8539641 | stable, FSDP1 ZeRO-2, **`ref_cpu_offload=true`**           | 61              |

Each row changes one or more variables compared to the prior; all crash at exactly step 62 with the same staircase signature.

**Definitively ruled out as cause**:
- Hyperparameters (lr, kl_coeff, warmup, epsilon)
- FSDP version (FSDP2 FULL_SHARD ≡ FSDP1 SHARD_GRAD_OP)
- Wsync cross-PG transport (XCCL/RDMA ≡ Gloo/TCP)
- Wsync frequency (every step ≡ every other step)
- Wsync presence (enabled ≡ disabled — proves wsync's XCCL pool is NOT the accumulator)
- Dataset content (batch 62 with `seed=42` is a routine 131-token GSM8K prompt; steps 61 and 63 are similar)
- Ref model placement (static XPU ≡ static CPU offload)

### What this isolates

The wall is in the **per-step training+ref forward FSDP-allgather loop itself**. Per step the recipe fires:
1. `generate_trajectory` → policy fwd (no-grad, skipped under single-epoch on-policy)
2. `generate_trajectory` → ref fwd (no-grad, full FSDP allgather across all params)
3. `grpo_step` → training fwd (grad, full FSDP allgather across all params)
4. `grpo_step` → bwd (FSDP reduce_scatter across all params)
5. `optimizer.step()`
6. `_sync_weights_to_vllm_xccl` (in normal runs) — XCCL broadcast, NOT FSDP allgather

Step 6 is innocent (job 8539607 disabled it and still hit step 62). Steps 2 and 3 are the two FSDP-allgather invocations of the policy params per training step. After ~62 such steps (~124 allgather cycles on the policy + ~62 on the ref), the XCCL/L0 driver wedges in one of the allgathers and the collective times out.

### Why BioReason survives 200 steps — still unexplained

BioReason (`docs/reports/bioreason_4b_200step_stability_20260501.md`) runs the same 2-node 11-train + 12-vLLM topology with FSDP1 SHARD_GRAD_OP and reaches 200/200 steps clean. Differences between recipes that might explain this gap, ordered by what's cheapest to test next:

1. **Activation checkpointing implementation**. BioReason calls `backbone.gradient_checkpointing_enable(use_reentrant=False)` (HF Trainer-style AC, recomputation triggered inside the HF module). The base recipe uses `training.set_activation_checkpointing(model, auto_wrap_policy={TransformerSelfAttentionLayer})` (PyTorch `torch.utils.checkpoint`). These produce different recompute graphs during bwd → different number of FSDP allgather calls per training step. Worth measuring the per-step allgather count directly.
2. **FSDP wrap target**. BioReason wraps the whole `BioReasonModel` (a thin nn.Module around the HF Qwen3-4B). Base recipe wraps the torchtune `TransformerDecoder`. Different sub-module decomposition → different `_fsdp_wrapped_module` boundary → different per-layer allgather sizes/counts.
3. **`disable_dropout(model)`**. Base recipe calls this after FSDP wrap (`grpo_full_finetune_distributed_xpu.py:1990, 2105`). BioReason does not. Probably innocuous but cheap to remove for a test.
4. **`use_orig_params=True` interaction with HF model layers**. Both use it but combined with different AC styles the bwd recompute may differ.

Each is a small, cheap A/B that should narrow the hypothesis space.

## What we tested tonight, in order

| Order | Change                                                  | Result vs step-62 wall |
|------:|---------------------------------------------------------|------------------------|
| 1     | Turn off `TORCHTUNE_VARLEN_NOGRAD_BYPASS`               | No effect on wall (fixed kl_loss explosion) |
| 2     | Add Q/K un-permute ENGAGED log                          | Cosmetic — confirms Llama fix engaging |
| 3     | `lr=5e-6`, `kl_coeff=0.02`, `warmup=10`                 | No effect on wall (fixed NaN cascade) |
| 4     | `vllm_weight_sync_interval=2`                           | No effect on wall |
| 5     | `WSYNC_CROSS_METHOD=gloo` + `GLOO_SOCKET_IFNAME=hsn0`   | No effect on wall (cleared a *different* wall) |
| 6     | `use_fsdp1_zero2: true` (new base-recipe flag)          | No effect on wall |
| 7     | `vllm_weight_sync: false` (disabled entirely)           | No effect on wall — wsync exonerated |
| 8     | `ref_cpu_offload: true`                                 | No effect on wall — CPU bounce hypothesis disproven |

## Shipped artifacts

### Code

- `recipes/dev/grpo_full_finetune_distributed_xpu.py`
  - New `_setup_model_fsdp1_flat_zero2` (FSDP1 SHARD_GRAD_OP flat wrap, mirrors BioReason). Gated by new `cfg.use_fsdp1_zero2` flag.
  - Q/K un-permute `ENGAGED` log lines on first sync in 4 wsync paths (colocate / ray-colocate / dedicated / xccl).
- `torchtune/dev/rl/weight_sync.py`
  - Broadened `use_fsdp1` gate (was `_use_fsdp1 AND dp_replicate>1`, now just `_use_fsdp1`) so the flat-FSDP1 case routes through `FSDP.state_dict()` instead of `.full_tensor()`.
  - Q/K un-permute wired into the FSDP1 XCCL wsync branch with its own ENGAGED log.
- `experiments/lora_grpo/run_qwen3_4b_dense_2node.sh`
  - Plumbed `WSYNC_CROSS_METHOD` / `WSYNC_INTRA_METHOD` / `GLOO_SOCKET_IFNAME` through both SSH heredocs so the AGPT launcher can override the wsync transport.

### Configs / launchers

- `recipes/configs/dev/production/auroragpt_2b_grpo_2n_gsm8k_server_xpu_stable.yaml` — lr=5e-6, kl_coeff=0.02, warmup=10, `use_fsdp1_zero2: true`, `ref_cpu_offload: true`.
- `experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_server_stable.sh` — submits the stable config with `WSYNC_CROSS_METHOD=gloo` default.
- `experiments/auroragpt_2b_bakeoff/pbs_2n_gsm8k_server_stable_no_wsync.sh` — diagnostic launcher that disables wsync.

### Memory entries

- `feedback_varlen_nograd_bypass_unsafe_on_unconverged_ref.md` — the rule.
- `feedback_emdash_breaks_ssh_heredoc.md` — em-dash inside `<<EOF` SSH heredoc bodies silently breaks dispatch on Aurora.
- `project_agpt2b_gsm8k_2n_run_diag_20260612.md` — full failure tree + 4 UPDATE blocks tracking the 7-run isolation.

### Tests

CPU regression suite (245 tests) still green; nine new Q/K un-permute round-trip tests cover the engaged path. DAPO/IS pre-existing failures unchanged (24 — for unimplemented features).

## 2026-06-12 late-session addendum: backward isolation

New env-gated diagnostics narrowed the step-62 wall beyond the original 7-run matrix:

| Job | Diagnostic | Result |
|-----|------------|--------|
| 8540095 | `TORCHTUNE_SKIP_REF_FWD=1`, `TORCHTUNE_SKIP_GRPO_STEP=1` | Generation-only passed `METRICS step=70`, exit 0 |
| 8540123 | `TORCHTUNE_SKIP_REF_FWD=1`, `TORCHTUNE_SKIP_GRPO_BACKWARD=1` | Policy train forward/loss-only passed `METRICS step=70`, exit 0 |
| 8540145 | `TORCHTUNE_SKIP_REF_FWD=1`, `TORCHTUNE_SKIP_GRPO_UPDATE=1` | Policy forward+backward reproduced wall: `grpo=175.8s` at step 61, `METRICS step=62`, then SIGABRT |
| 8540248 | Same as 8540145 plus `TORCHTUNE_GRPO_BACKWARD_NO_SYNC=1` | FSDP `no_sync()` still reproduced wall after `METRICS step=62`; `grpo=31.9s` at step 61 |
| 8540279 | Same as 8540248 plus `enable_activation_checkpointing=false` | Worse: step-1 SIGABRT, matching earlier AC-off behavior |
| 8540432 | `FORWARD_BATCH_SIZE=8` but default `TORCHTUNE_USE_CHUNKED_LOSS=1` | Still single-backward path; reproduced wall after `METRICS step=62` |
| 8540465 | `TORCHTUNE_USE_CHUNKED_LOSS=0`, FBS=8 | Exposed missing FSDP1 metadata init in dormant chunked path |
| 8540485 | Same as 8540465 after setting `_fsdp2_param_groups_meta=[]` for FSDP1 | True chunked FBS=8, update skipped, passed `METRICS step=70`, exit 0 |
| 8540537 | True chunked FBS=8, optimizer/wsync enabled, ref still skipped | Passed `METRICS step=70`, exit 0 |
| 8540596 | Near-production true chunked FBS=8, real ref forward restored | Passed `METRICS step=70`, exit 0 |
| 8540637 | Stable defaults promoted to true chunked FBS=8, `NSTEPS=150` | Crossed old wall; failed after `METRICS step=80` with GPU `banned:1` |
| 8540680 | Same as 8540637 but `lr=0`, `kl_coeff=0`, `NSTEPS=100` | Also failed after `METRICS step=80` with GPU `banned:1`; proves new wall is resource/driver, not learning dynamics |
| 8540757 | Same as 8540680 but `TORCHTUNE_SKIP_REF_FWD=1` | Passed `METRICS step=100`, exit 0; proves step-80 wall requires real ref forward |
| 8540785 | Same as 8540680 but `ref_forward_batch_size=8` | Passed `METRICS step=100`, exit 0; proves ref forward peak shape/chunking is the step-80 lever |
| 8540864 | Promoted stable defaults: train FBS=8, ref FBS=8, `TORCHTUNE_USE_CHUNKED_LOSS=0`, `NSTEPS=150` | Passed `METRICS step=150`, exit 0; production workaround proven |

This changes the most precise root-cause boundary:

- Not vLLM generation.
- Not ref forward itself.
- Not optimizer, LR scheduler, or weight sync.
- Not FSDP grad sync/reduce-scatter alone, because `no_sync()` still fails.
- Required component in the failing path: policy backward executed as one large single-backward over all 16 GRPO samples.
- True chunked FBS=8 changes the backward shape and clears the old step-62 wall through step 70.
- With real ref forward at chunk=16 and optimizer/wsync restored, the new one-process wall is around step 80 and manifests as GPU `banned:1`.
- The same step-80 `banned:1` occurs with `lr=0` and `kl_coeff=0`, so the new wall is resource/driver accumulation, not learning dynamics.
- Skipping ref forward passes 100 steps, and chunking ref forward to 8 also passes 100 steps; the step-80 wall is therefore ref-forward peak shape/resource pressure.
- Combining train FBS=8, ref FBS=8, and `TORCHTUNE_USE_CHUNKED_LOSS=0` passes the full 150-step target in one process (job 8540864).

Potential workaround noted but not yet tested: a replicated/DDP-style training path for AGPT-2B on 11 tiles may avoid FSDP backward unshard/reshard entirely if HBM allows. Treat this as a fallback topology workaround after exhausting the now-promising true chunked-backward path, because it changes the training memory envelope and distributed semantics more than the current env-gated probes.

## Recommendations / next session

1. **Use train FBS=8, ref FBS=8, and `TORCHTUNE_USE_CHUNKED_LOSS=0` as the AGPT-2B 2N production envelope.** Job 8540864 validated the full 150-step target in one process.
2. **Keep checkpoint/restart as fallback only**, not the primary path. If future runs extend beyond 150 steps and hit a new wall, segment at 100-150 steps using the same envelope.
3. **Instrument per-step FSDP allgather/backward collective count**. Add a counter wrap around FSDP collectives and dump per-step counts on rank 0. Compare AGPT-2B vs BioReason per-step counts to quantify why train/ref chunking clears the walls.
4. **Keep DDP-style replicated training as an explicit fallback**. If a longer horizon still hits an XPU driver wall, test a DDP/replicated policy path for AGPT-2B if HBM permits.
5. **Submit to Intel/ALCF as a refinement of `UPSTREAM_FILING_DRAFT_l0_resource_pool.md`**. AGPT-2B's deterministic step-62/80 walls are new data points on top of the existing 4 sig#2/4 reproducers — same family but a fresh sub-mechanism (the per-iteration count is decoupled from explicit `empty_cache()` calls; our code never makes them on XPU).

## Related

- `docs/bugs/intel_xpu_resource_leak_bug_report.md` — sig#2 canonical reproducer (FSDP + empty_cache + storage.resize_).
- `docs/bugs/UPSTREAM_FILING_DRAFT_l0_resource_pool.md` — combined L0 leak family draft.
- `docs/reports/cxi_mr_step28_crash_investigation_20260428.md` — the 32B step-28 wall (different mechanism, CXI MR cache).
- `docs/reports/bioreason_4b_200step_stability_20260501.md` — the comparison case that survives 200 steps.
- `memory/project_agpt2b_gsm8k_2n_run_diag_20260612.md` — the running failure tree (this report's source).
