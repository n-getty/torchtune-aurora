# Qwen3-32B Dense GRPO — Multinode Exemplar (Aurora XPU)

The canonical, runnable Qwen3-32B GRPO setup for production runs on Aurora.

## Topology

3 PBS nodes:
- **Node 0** — vLLM only: 12 tiles → 3 vLLM replicas × TP=4.
- **Nodes 1-2** — training only: 12 tiles each → 24-way pure FSDP
  (`dp_replicate=1`).

Weight sync is gloo cross-PG (CPU/TCP over `hsn0`) + XCCL intra, every
2 steps (deferred). Gloo cross eliminates the CXI MR cache leak that caps
XCCL-cross runs at ~30 steps.

## Files

| Role | Path |
|---|---|
| Launcher | `recipes/dev/run_qwen3_32b_grpo_3node.sh` |
| Learning config (G=8 mg=1024) | `recipes/configs/dev/production/qwen32B_grpo_3node_24way_xpu.yaml` |
| Stable learning config (G=8 mg=512) | `recipes/configs/dev/production/qwen32B_grpo_3node_24way_stable_xpu.yaml` |
| Recipe | `recipes/dev/grpo_full_finetune_distributed_xpu.py` |

## Hold a job

```bash
qsub -I -l select=3 -l walltime=1:00:00 -q debug-scaling -A <project>
```

## Default launch (stable envelope)

```bash
export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
bash recipes/dev/run_qwen3_32b_grpo_3node.sh
# Reads recipes/configs/dev/production/qwen32B_grpo_3node_24way_stable_xpu.yaml
# G=16 fbs=16 max_gen=128, ~41s/step (status.md Test A).
```

## Throughput launch

```bash
export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
NSTEPS=20 \
  CONFIG=recipes/configs/dev/production/qwen32B_grpo_3node_24way_xpu.yaml \
  bash recipes/dev/run_qwen3_32b_grpo_3node.sh
# G=32 fbs=16 max_gen=128, ~53s/step (1.54× per-sample throughput vs G=16).
```

## Sweeping G/fbs/max_gen without editing the YAML

The launcher only emits `grpo_samples=` / `forward_batch_size=` /
`max_generated_tokens=` CLI overrides when the corresponding env var is set.
Examples:

```bash
GRPO_SAMPLES=24 bash recipes/dev/run_qwen3_32b_grpo_3node.sh
MAX_GEN_TOKENS=192 bash recipes/dev/run_qwen3_32b_grpo_3node.sh   # marginal
```

Otherwise the YAML's value wins.

## Required env (set by the launcher)

| Var | Value | Why |
|-----|-------|-----|
| `CCL_PROCESS_LAUNCHER` | `none` | SSH-launched torch.distributed.run, no PALS context |
| `CCL_ATL_TRANSPORT` | `ofi` | matches `none` launcher |
| `WSYNC_CROSS_METHOD` | `gloo` | avoids CXI MR cache leak ("FSDP collectives" RC, status.md) |
| `WSYNC_INTRA_METHOD` | `xccl` | 2.4× faster than gloo intra |
| `GLOO_SOCKET_IFNAME` | `hsn0` | gloo cross-PG bandwidth (1.3 GB/s vs ~0.1 GB/s on lo) |
| `TORCHTUNE_USE_CHUNKED_LOSS` | `1` | per-chunk fwd+bwd; required to avoid OOM |
| `TORCHTUNE_PINNED_CPU_BUF` | `1` | gather 31s → 3.7s (8.5× speedup) |
| `PYTORCH_ALLOC_CONF` | `max_split_size_mb:512,garbage_collection_threshold:0.6` | Documented stable for 24-way |
| `CCL_WORKER_COUNT` | `1` | `4` causes 48× AllGather regression |
| `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` | `65536` | default 1000 → IPC eviction → `banned:1` |

## Expected timings

The mg=128 rows are **memory-pressure smokes** — Qwen3-32B's GSM8K responses
need >128 tokens to close `<answer>` tags, so rewards collapse to ~0 at
mg=128 even though step time is fast. The mg=512/1024 rows at G=8 are the
**learning envelopes** — actual reward signal and KL evolution.

| Variant | step | wsync | result |
|---|---|---|---|
| **EXEMPLAR — Smoke E, G=8 fbs=8 mg=1024** (2026-05-01) | **~85s** | ~0.3s | 5/5 clean, rewards 76-101, resp_len 258-754, KL 0.0005-0.0009, resv 59 GiB |
| Smoke D, G=8 fbs=8 mg=512 (2026-05-01) | ~67s | ~0.3s | 10/10 clean, rewards 0.1-101 (cap collapse on hard prompts at 511 tok) |
| Smoke C, G=8 fbs=8 mg=128 (2026-05-01) | ~28s | ~0.3s | 5/5 clean but rewards ≈0 (every response truncated before `</answer>`) |
| Memory-smoke G=16 fbs=16 mg=128 | ~41s | ~0.3s | Test A 5/5 clean (no learning signal — superseded) |
| Memory-smoke G=32 fbs=16 mg=128 | ~53s | ~0.3s | Test B 5/5 clean (no learning signal — superseded) |
| Memory-smoke G=32 fbs=16 mg=192 | ~72s | ~0.3s | Test G2 3/3 clean (1.50 GiB free, marginal — superseded) |
| Memory-smoke G=16 fbs=16 mg=512 | ~82s | ~0.3s | Test D 3/3 clean (0.01 GiB free, absolute limit — superseded) |
| 2-node 2-hop XCCL, G=16 fbs=4 mg=128 | ~43s | 9.1s | 24/24 clean (job 8450367) |
| 2-node 2-hop **gloo cross**, G=16 fbs=4 mg=128 | ~67s | 47s + 32s wait | 20/20 clean (Run 8) — CXI leak eliminated |

## Dataset choice — GSM8K is saturated by Qwen3-32B base

Smoke E (G=8 mg=1024) on GSM8K showed `reward_mean=101.0  std=0.000` for
4/5 steps with `min=max=101`. That's not a learning curve — it's
Qwen3-32B base nailing every easy GSM8K problem. The 32B GRPO runtime is
genuinely working (ratios=1.0000, KL evolves, gradients flow on step 3
when responses varied), but **GSM8K does not exercise reward-driven
improvement on a 32B-class base model**.

For future infra/algorithm validation, pick a dataset with a non-trivial
accuracy ceiling for Qwen3-32B base. The MATH benchmark (Hendrycks et al.)
is a standard choice — no `torchtune.dev.rl.math_dataset` module exists
yet; add one paralleling `torchtune/dev/rl/gsm8k.py` when the next harder
benchmark is needed.

## Reward functions

The recipe honors `cfg.reward_functions` for the math `reward_mode` (the
default). When declared, each `Reward` instance is invoked per-completion
and the per-function rewards are summed into the trajectory reward.
Production YAMLs use:

- `TaggedMathCorrectnessReward` — parses `<answer>...</answer>` then runs
  `math_verify`. Defaults: `100` exact, `50` substring match, `1` non-empty
  answer, `0` empty/missing.
- `ThinkingTagPresenceReward` — `1` if `<think>` content is non-empty, else `0`.

A response that is correct AND has a `<think>` block scores `100 + 1 = 101`,
which matches the `BATCH_REWARD` lines in Smoke E. When `cfg.reward_functions`
is absent, the recipe falls back to the legacy hardcoded
`batched_rewards()` path with the same numerical semantics.

## Hard envelope (do not exceed without a memory probe first)

At the current learning envelope (G=8/mg=1024):
- `resv` saturates at ~59 GiB (5 GiB headroom on 64 GiB tile).
- `mg=1024` is the validated ceiling. `mg=1536` not yet measured — likely
  tight on memory.

At higher G (memory-smoke envelopes G≥16, mg=128) the historical limits hold:
- `G/fbs <= 2` — exceeding 2 forward chunks triggers CCL external memory
  explosion (1.85 → 13–15 GiB), hangs at step 1.
- `max_generated_tokens` — `128` safe, `192` marginal (1.5 GiB free
  POST-BWD at G=32), `>=256` OOMs at G=16/G=32.
- `G=48+` — hangs (3-chunk territory).
- `G=64` — XPU kernel indexing bug in `batched_logits_to_logprobs`, not OOM.

## 2-node fallback

If only 2 nodes are available, the working `experiments/multinode_32b/run_32b_2hop_production.sh`
script (untracked; lives in the local working tree) runs a 1+1 topology with
XCCL 2-hop weight sync. Faster per-sync (9.1s) but CXI MR pressure caps useful
run length to ~30 steps. Not a flagship target — the 3-node path above is
what should be reproduced for benchmarks.

## IPEX `varlen_attention` (XPU fast path)

`TORCHTUNE_USE_IPEX_VARLEN=1` routes causal-only SDPA through IPEX and gives
a 19% step speedup + 5 GiB lower steady-state reserved on **BioReason** (see
`docs/reports/bioreason_ipex_varlen_20260430.md`).

**On dense Qwen3-32B GRPO this flag silently does nothing in the standard
path** — the recipe builds an explicit causal mask in `generate_trajectory()`
and passes it to policy / ref / training forwards, and the varlen gate
requires `mask is None`. To confirm engagement, grep the worker log for the
one-shot `varlen=` line emitted from `torchtune/modules/attention_utils.py`:

- `varlen=engaged` — IPEX path active.
- `varlen=requested-but-skipped (mask is not None)` — env var set, dense
  Qwen3 normal path. **This is expected for 32B GRPO today.**
- `varlen=disabled (TORCHTUNE_USE_IPEX_VARLEN unset)` — env var not set.

Removing the explicit mask for dense 32B (to enable varlen) is a worthwhile
experiment but must preserve response padding semantics; not done here.

## Related

- Status: `docs/status.md` "3-Node 24-Way FSDP for Qwen3-32B" section
- IPEX gate: `torchtune/modules/attention_utils.py:269-277`
- Per-test memory + crash analysis: `memory/project_chunked_loop_regression.md`,
  `memory/project_gloo_cross_pg_fix.md`, `memory/project_pinned_cpu_buffer.md`.
