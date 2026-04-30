# TorchTune — Aurora XPU Fork

This is a fork of [pytorch/torchtune](https://github.com/pytorch/torchtune) adapted for
**Aurora HPC** (Intel Max Series GPUs / XPU), focused on reinforcement learning (GRPO) for
large language models and multimodal systems.

[Getting Started](docs/getting_started.md) | [Current Status](docs/status.md) | [CLAUDE.md](CLAUDE.md) | [Upstream torchtune](https://github.com/pytorch/torchtune)

---

## Platform

Aurora is an Intel GPU cluster at Argonne National Laboratory. Each node has 6 Intel Max 1550
GPUs presenting as **12 XPU tiles**, each with 64 GiB HBM. Interconnect is Slingshot 11.
The distributed backend is Intel oneCCL (XCCL), not NCCL.

Key hardware facts relevant to this work:
- 64 GiB HBM per tile — enables single-node 32B and MoE models infeasible on A100-40GB
- XeLink for intra-node tile communication (~10 GB/s)
- Slingshot 11 for inter-node (~25 GB/s per link)
- Level Zero (L0) driver underneath PyTorch XPU

---

## Key Adaptations

### XPU / oneCCL backend

All `torch.cuda` references translated to `torch.xpu` or device-agnostic equivalents.
Distributed training uses oneCCL via the XCCL backend with specific env var requirements
(see [CLAUDE.md](CLAUDE.md) and the PBS template in [docs/getting_started.md](docs/getting_started.md)).

### GRPO recipe architecture

The recipe layer is split into a general base and model-specific subclasses:

```
GRPOFullFinetuneDistributedXPU   (recipes/dev/grpo_full_finetune_distributed_xpu.py)
└── GRPOBioReasonDistributedXPU  (recipes/dev/grpo_bioreason_distributed_xpu.py)
```

Heavy RL infrastructure lives in `torchtune/dev/rl/` and is bound to the recipe via explicit
method injection (not inheritance). When you see `self._foo()` in the recipe without a local
`def _foo`, look in `torchtune/dev/rl/*.py`.

New model recipes subclass `GRPOFullFinetuneDistributedXPU` and override `setup()`,
`generate_trajectory()`, and `grpo_step()`. See `grpo_bioreason_distributed_xpu.py` as the template.

### Weight sync for vLLM rollouts

Three modes, selected by `vllm_mode` config key:

| Mode | Transport | Latency | Best for |
|------|-----------|---------|----------|
| `colocate_sleep` | Shared memory (SHM) | ~1.4s async, hidden in gen | 3B single-node |
| `server` (2-hop XCCL) | gloo cross-PG + XCCL intra-PG | ~9s for 32B | 32B multi-node |
| `server` (shared-FS) | `/lus/flare/` raw bytes + `/collective_rpc` | ~15s | BioReason multimodal |

The 2-hop XCCL design: training rank 0 → vLLM TP-rank 0 via a 2-rank gloo group over
Slingshot, then vLLM TP-rank 0 → all TP workers via XCCL intra-node. This avoids the
flat broadcast bottleneck (1.7 GB/s for 12 receivers vs 8 GB/s for the 2-rank cross hop).

### MoE expert forward (BMM kernel)

`torchtune/modules/moe/experts.py` implements scatter-pad-BMM-gather replacing sequential
per-expert matrix multiply: **6.3× speedup** (30.14s → 4.75s fwd+bwd on 6 tiles) for
Qwen3-30B-A3B and Gemma4-26B-A4B.

### BioReason multimodal GRPO

`GRPOBioReasonDistributedXPU` integrates an ESM3 protein encoder + GO term encoder +
Qwen3-4B LLM backbone. Multimodal prompt embeddings are generated on the training side
and shipped to vLLM via HTTP (`prompt_embeds` extension), enabling GRPO over protein
sequences with gene ontology (GO term F1) reward.

Architecture: 11 training tiles (FSDP1 SHARD_GRAD_OP) + 12 dedicated vLLM tiles (HTTP
server DP=12) across 2 nodes. Weight sync via shared filesystem raw bytes.

### Async GRPO

`torchtune/dev/rl/async_rollout.py` (`RolloutProducer`) overlaps vLLM generation with
the training backward pass. Phase 1 validated on Qwen3-3B: **22.6s/step**, IS-corrected
loss, importance ratios ≈ 1.000×, producer fully overlapped with GRPO backward.

---

## Validated Models & Performance

All results on Aurora (Intel Max 1550, 64 GiB/tile), with weight sync enabled.

| Model | Config | Nodes | Step time | Status |
|-------|--------|-------|-----------|--------|
| Qwen2.5-3B | 10+2 tiles, SHM sync, G=16 | 1 | ~21 s | Production-ready: 130 steps clean |
| Qwen3-3B | async GRPO (RolloutProducer), G=8 | 1 | ~23 s | Phase 1 validated |
| Qwen3-30B-A3B (MoE) | 10+2 tiles, SHM sync, G=8 fbs=8 | 1 | ~55 s | 3/3 steps, 9.2 tok/s |
| Qwen3-32B | 3-node 24-way FSDP, G=16 fbs=16 | 3 | **~41 s** | 5/5 clean, pinned CPU buffer |
| Qwen3-32B | 3-node 24-way FSDP, G=32 fbs=16 | 3 | **~53 s** | 5/5 clean, 1.54× per-sample vs G=16 |
| Qwen3-32B | 2-node dedicated vLLM, 2-hop XCCL | 2 | ~43 s | 24/24 clean (short runs, <30 steps) |
| Gemma4-26B-A4B | 10+2 tiles, server mode, G=16 | 1 | ~24 s | Infrastructure validated |
| BioReason-Pro 4B | 2-node server mode, 11+12 tiles | 2 | ~58 s | 5/5 clean, KL evolving |

**Performance context**: Aurora (torchtune) at ~41s/step for 32B G=16 vs H100 NVL
(same code) at ~15s/step. The 64 GiB tile enables 32B at batch sizes infeasible on
A100-40GB.

See [docs/status.md](docs/status.md) for full run history and current baselines.

---

## Works in Progress

| Area | State | Blocker |
|------|-------|---------|
| Expert Parallelism (Qwen3-30B-A3B, EP=4) | Forward works across 12 ranks | Train fwd L0 IPC handle pressure causes banned:1 PDE crash |
| Expert Parallelism (Gemma4-26B-A4B, EP=4) | Forward + backward working (v161) | Production validation pending |
| Async GRPO Phase 2 | Dedicated-rank async design WIP | — |
| Gemma4-31B gene recall | Infrastructure stable | Training stability: SFT warm-up + EOS fix needed |
| 3-node DP>1 | Per-replica PGs implemented | Validation pending |

---

## Repository Layout (Key Files)

```
recipes/dev/
  grpo_full_finetune_distributed_xpu.py    # General GRPO base recipe (~3750 lines)
  grpo_bioreason_distributed_xpu.py        # BioReason subclass (~1450 lines)
  async_grpo_full_finetune_distributed.py  # Async GRPO (RolloutProducer)
  run_grpo_vllm_xpu.sh                     # Single-node vLLM launcher (3B/8B/30B)
  run_gemma4_grpo_vllm.sh                  # Gemma4 single-node launcher
  run_qwen3_30b_ep4_vllm_3node.sh          # Qwen3-30B EP=4 3-node launcher

torchtune/dev/rl/
  weight_sync.py                           # All weight sync runtime (~1770 lines)
  vllm_backend.py                          # vLLM init and mode setup (~690 lines)
  distributed.py                           # _slice_trajectory, _gather_trajectory, init_xpu_pg
  loss.py                                  # GRPOSimpleLoss, GRPOLoss, chunked fwd/bwd
  rewards.py                               # math_reward_fn, gene_recall_reward_fn
  async_rollout.py                         # RolloutProducer
  types.py                                 # GRPOTrajectory, GRPOStats

torchtune/dev/bioreason/
  model.py                                 # BioReasonModel (ESM3 + GO + Qwen3-4B)
  dataset.py                               # bioreason_rl_dataset
  reward.py                                # bioreason_reward_fn (GO-term F1)

torchtune/modules/moe/
  experts.py                               # GroupedExperts (BMM scatter-pad-gather)
  _parallelism.py                          # EP AllToAll dispatch/combine

experiments/
  multinode_32b/    run_32b_3node_24way.sh, run_32b_2hop_production.sh, ...
  bioreason/        hold_bioreason_2node.sh, ...
  gene_recall/      hold_gene_recall_prod.sh, ...
  async_grpo/       run_phase1_async.sh, ...
  ep_parallelism/   hold_qwen3_ep_*.sh, ...

recipes/configs/dev/production/
  qwen3B_grpo_colocate_xpu.yaml
  qwen3_30b_a3b_grpo_xpu.yaml
  qwen32B_grpo_server_xpu.yaml
  bioreason_4b_grpo_2node_server_xpu.yaml
  gemma4_26b_a4b_grpo_server_xpu.yaml
  ... (and more)

docs/
  getting_started.md       # Environment setup, launch commands
  status.md                # Current experiment state and baselines
  bugs/                    # Specific bug investigations
  features/                # vllm_weight_sync.md, moe_integration.md
  reports/                 # Per-experiment reports
```

---

## Quick Start

See [docs/getting_started.md](docs/getting_started.md).

---

## Syncing with Upstream

This repo tracks [pytorch/torchtune](https://github.com/pytorch/torchtune) as the `upstream`
remote. Only 4 upstream files are modified (backend string, XPU util exports, minor fixes),
so rebases are typically clean.

```bash
git fetch upstream
git rebase upstream/main
```

---

## License

Original torchtune is released under the [BSD 3 license](./LICENSE).
