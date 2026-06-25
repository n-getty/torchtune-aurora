# CLAUDE.md — TorchTune for Aurora/XPU

## Goal

Adapt TorchTune's RL recipes (PPO, GRPO) to run on **Aurora HPC** (Intel Max Series GPUs / XPU). Primary focus: GRPO training for large language and multimodal models using vLLM for rollout generation. The specific active workstreams shift over time — see the dated **Current Status** section below and `docs/status.md` for what is under active iteration vs. validated-but-paused.

## Critical Platform Constraints (Aurora)

- **XPU, not CUDA** — all `torch.cuda` calls must become `torch.xpu` / device-agnostic
- **oneCCL backend** (via XCCL) — not NCCL. Requires specific env vars (see below)
- **`torch.compile`** — viable single-node AND multi-node (backbone-only compile). The historical multi-node deadlock was the `CCL_WORKER_COUNT=4` + `CCL_REDUCE_SCATTER=ring` bug fixed Apr 2026. Validated 2026-06-10 on 2N microbench (no hang) AND on dense 4B GRPO 2N at production envelope (fbs=2, ref_fbs=16): eager 38.1s/step vs compile 38.2s/step — **wash**, `ratios=1.0000` bit-exact. grpo_step is FSDP-collective dominated (~21s of ~38s), and compile doesn't speed up allgather/reduce-scatter. Compile is more likely to pay off in colocate-mode (no cross-rank FSDP fwd+bwd collectives) or on dense 32B (training overhead grows faster than per-collective startup). Memory pressure: dense 4B at fbs=2 OOMs in bwd under compile — drop to fbs=1 or set `TORCHTUNE_USE_CHUNKED_LOSS=1`. See `memory/project_torch_compile_multinode_xpu_unblocked.md`.
- **`glob.glob()` hangs** on DAOS/dfuse mounts — use `os.listdir()` + filtering
- **No `device_id` in `init_process_group`** — causes DataLoader worker hangs on XPU
- **`ZE_AFFINITY_MASK=$LOCAL_RANK`** — must be set to prevent 144 L0 device contexts
- **FSDP per-module wrapping** causes catastrophic overhead on XPU — use top-level-only wrapping
- **`torch.xpu.empty_cache()` + FSDP** leaks UR handles in Level Zero — NEVER call `empty_cache()` in FSDP training loops. See `docs/bugs/intel_xpu_resource_leak_bug_report.md`. NOTE: the `force_math_sdpa` config flag (which calls `torch.backends.cuda.enable_flash_sdp(False)`) is **a no-op on XPU** — those toggles affect the CUDA dispatcher only. Validated 2026-04-30: identical timing and peak memory regardless of the flag. Don't vary it expecting behavioral change on XPU.
- **`torch.xpu.empty_cache()` + XCCL list-style `all_gather`** leaks device-global memory (separate from the FSDP UR-handle leak above). Upstream: [Aurora #143](https://github.com/argonne-lcf/AuroraBugTracking/issues/143) / [torch-xpu-ops #3744](https://github.com/intel/torch-xpu-ops/issues/3744). `ProcessGroupXCCL` allocates a hidden flat `newLikeFlat()` temp inside list-style `dist.all_gather(list_out, local, group)`; when `empty_cache()` later runs, L0 fails to reclaim it. Drop per iter exactly equals `(world_size − 1) × tensor_bytes`. **Workaround:** use `dist.all_gather_into_tensor(flat_out, local, group)` with a pre-allocated persistent buffer — the caller owns the output, no hidden temp. All EP wsync per-projection AGs in `torchtune/dev/rl/weight_sync.py` have been converted (bf16 baseline + fp8-wire). Test: `tests/torchtune/dev/rl/test_ep_wsync_into_tensor_equivalence.py`. Diagnostic signature for new occurrences: `mem_get_info` free drops by `(W−1)×bytes/iter` while `memory_stats` returns to baseline. **Upstream status (2026-06-01):** torch-xpu-ops #3744 was *closed* by Intel — pkourdis reproduced the leak with our reproducer but routed it as oneCCL-side (internal Intel JIRA `MLSL-4397`, not publicly trackable). Closure means the report was received and forwarded — *not* that the leak is fixed. No public fix ETA; keep the `all_gather_into_tensor` workaround indefinitely.
- **Colocate TP=8**: Always use `vllm_enforce_eager=True` — PIECEWISE is 75% slower (72 XCCL crossings/step overwhelm the 7.6% graph-replay gain). XPU graph + enforce_eager fixes are in `xpu.py` and `vllm_backend.py`. Re-run ocloc pre-compile (`/usr/bin/ocloc`, target `pvc`) when switching torch211 venv — stale SPIR-V cache causes 50+ min first-call hang. See `memory/project_colocate_tp8_triton_zebin_fix.md`.
- **Colocate YAML**: Always set `reshard_after_forward: true` — the old ZeRO-2 default combined with vLLM's own weight copy pushes alloc to 55+ GiB and causes CCL stalls.
- **`gen_batch_size`**: Defaults to `batch_size` (one vLLM call/step). Set to `forward_batch_size` only if vLLM OOMs on the full batch.
- **`ref_forward_batch_size` sharp edge**: defaults to `forward_batch_size`. ref forward is no-grad — there's no activation-checkpointing cost forcing it to match training fbs. If you drop `fbs` to 1 (e.g. for compile memory headroom) and DON'T set `ref_forward_batch_size` explicitly, the recipe runs `num_seqs` sequential FSDP-allgather ref-fwd cycles. Validated 2026-06-10: dense 4B 2N G=4 fbs=1 ref_fwd inflated 0.2s → 100s (500×). **Always set `ref_forward_batch_size: 16` (or ≥ `grpo_samples × batch_size`) in YAML, or pass `REF_FORWARD_BATCH_SIZE` from launcher.** See `memory/feedback_ref_forward_batch_size_default_trap.md`.

### Launcher decision table (authoritative)

The repo supports three distinct launch scopes. Mixing CCL/affinity envs across them silently breaks training. Pick one row and follow it; never inherit env from another row.

| Scope                                | Launcher                          | `CCL_PROCESS_LAUNCHER` | `CCL_ATL_TRANSPORT` | `CCL_KVS_MODE` | `ZE_AFFINITY_MASK`            | `init_process_group(device_id=)` |
|--------------------------------------|-----------------------------------|------------------------|----------------------|-----------------|-------------------------------|----------------------------------|
| Interactive single-node (held node)  | `torchrun --standalone`           | `none`                 | `ofi`                | (unset)         | unset (CCL needs all UUIDs)   | `xpu:LOCAL_RANK` (single-node only) |
| Production multi-node                | `mpiexec --pmi=pmix`              | `pmix`                 | `mpi`                | `mpi`           | set per `LOCAL_RANK`          | None (XPU multi-node hangs with `device_id`) |
| vLLM-only process (server/dedicated) | `torchrun` or direct `python`     | `none`                 | `ofi`                | (unset)         | set to isolated tile range    | n/a (vLLM owns its PG)           |

Notes:
- **Interactive single-node**: `pmix` + `mpi` envs hang `torch.distributed.run --standalone` for ~15 min on init (see memory `feedback_pmix_envs_break_standalone.md`). Use the row above.
- **Production multi-node**: requires the full env block in the next section AND `from mpi4py import MPI; MPI.COMM_WORLD.Barrier()` before `init_process_group()`.
- **vLLM-only**: TP=1 must use `ofi`/`none` — `mpi`/None silently kills the EngineCore spawn worker (see `feedback_vllm_tp1_ccl_env.md`).
- The recipe (`grpo_full_finetune_distributed_xpu.py:35-56` and `xpu_utils.py:126-150`) implements both training rows; the doc here is the contract.

### Required CCL Environment Variables (Multi-Node)
```bash
# MPI transport (official Aurora recommendation, 2x AllGather bandwidth vs ofi)
export CCL_PROCESS_LAUNCHER=pmix          # Requires: mpiexec --pmi=pmix
export CCL_ATL_TRANSPORT=mpi              # MPI-based transport (topology-aware)
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536  # CRITICAL: default=1000 evicts IPC handles → banned:1. But 65536 accumulates 10.85 GiB IPC handle memory by step 1 BWD on 10-tile FSDP2 → OOM at step 2. Single-node 32B 10+2 BLOCKED — use 2-node HSDP.
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi                    # Slingshot 11 interconnect
export CCL_WORKER_COUNT=1                 # CRITICAL: 4 causes 48x AllGather regression
export CCL_ALLREDUCE=ring                 # Ring for large tensor AllReduce
# NOTE: Do NOT set CCL_REDUCE_SCATTER=ring — causes 63x regression on multi-node
export CCL_CHUNK_SIZE=16777216            # 16MB chunks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT      # 12 tiles/node
```
**Note**: Requires `from mpi4py import MPI; MPI.COMM_WORLD.Barrier()` before `init_process_group()` in Python.

## Running on Compute Nodes

General HPC iteration discipline (hold-node not one-shot batch, never touch jobs you
did not submit, build the nodefile from real `exec_host`, two-monitor rule for hangs)
lives in the global skills — **`hpc-iteration-discipline`, `pbs`, `aurora`,
`monitor-long-jobs`, `launcher-config-drift`, `shell-quoting-traps`,
`python-env-shadowing-hpc`**. Invoke those for the how-to; this section keeps only the
Aurora/XPU-specific deltas that those skills do not cover.

**`.bashrc` auto-cd trap (Aurora-specific):** The login node `.bashrc` has a
`[[ $- == *i* ]] || return` guard and its auto-cd silently does nothing for
non-interactive SSH sessions (e.g. `ssh node cmd`). **Use absolute paths everywhere** —
scripts, configs, output files. Even a `cd /lus/... &&` prefix does not save a script
that uses relative paths internally if the SSH cd silently no-ops.

```bash
# WRONG — relative path fails if cd silently no-ops
ssh x4615c2s3b0n0 "cd /lus/flare/projects/ModCon/ngetty/torchtune && bash recipes/dev/run.sh"
# RIGHT — absolute path, works regardless of CWD (pass full paths as args too)
ssh x4615c2s3b0n0 "bash /lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/run.sh"
```

## Repository Layout (Key Files)

### XPU GRPO Recipes (primary development targets)

The recipe layer is split into a general base and model-specific subclasses:

- `recipes/dev/grpo_full_finetune_distributed_xpu.py` — **general GRPO base** (~5500 lines)
  - Training loop, FSDP setup, vLLM colocate/server/dedicated modes, weight sync dispatch, checkpointing
  - All method calls in this file are present either as `def` here or as **injected methods** from `torchtune/dev/rl/` — see below
- `recipes/dev/grpo_bioreason_distributed_xpu.py` — **BioReason subclass** (~1450 lines)
  - Subclass of `GRPOFullFinetuneDistributedXPU`; overrides `setup()`, `generate_trajectory()`, `grpo_step()`, `train()`
  - Adds `_setup_bioreason_models()` (loads ESM3 + GO + Qwen3-4B) and `_setup_bioreason_vllm_rank()` (dedicated vLLM tile)
- `recipes/dev/lora_grpo_full_finetune_distributed_xpu.py` — **LoRA-GRPO** (~2100 lines)
  - `LoRAGRPODistributedXPU(FTRecipeInterface)`; adapter training + merged-weight publish to vLLM (`_gather_merged_lora_weights` → `_publish_lora_to_vllm`)
  - **Three publish paths** (`lora.publish_mode`, supersedes the legacy `lora.use_runtime_lora` toggle):
    - **`merged`** (= old Path A, `use_runtime_lora=false`): merges `W_eff = W_base + (α/r)BA` and ships the **FULL ~6.77 GiB** (4B) via raw_bytes → `/collective_rpc load_weights_from_raw` every step — *not* a cheap adapter, sidesteps the `--enable-lora` PDE. Bandwidth-bound + on the critical path (~28s/step at 4B/2N: gather ~15s + join ~13s).
    - **`delta`** (Path C, merge-at-receiver — the optimal path, **HW-validated 2026-06-18**): ships the frozen base **ONCE** (~6.77 GiB) then only the **~66 MB** `lora_a`/`lora_b` each step; the vLLM worker re-merges `W_eff = base + scale·(B@A)` from a base cached **on the resident XPU device** (`load_lora_base_from_raw` / `load_lora_delta_from_raw` in `vllm_weight_sync_worker.py`). **Bit-exact to `merged`** (routes the assembled `W_eff` through the same `model.load_weights()`), no `--enable-lora`, any TP, stays on `frameworks/2025.3.1`. Q/K unpermute is applied sender-side to both base and delta (commutes with the add). **The merge MUST run on XPU** — a CPU merge measured 20.5s/step (252 fp32 matmuls × 12 tiles); on-XPU it's 0.16s. Fallback `TORCHTUNE_LORA_DELTA_BASE_CPU=1` if a tile lacks ~6.77 GiB free HBM. **A/B result (Qwen3-4B 2N G=8): step ~26.7s vs merged ~52s (~2× faster), publish join ~0.47s vs ~13s, both GREEN.** See `docs/reports/lora_delta_publish_path_20260617.md`. CPU pin-down: `tests/torchtune/dev/rl/test_lora_delta_{receiver_equivalence,sender_format}.py`. YAML default is still `merged` pending a ≥50-step soak.
    - **`runtime`** (Path B, = old `use_runtime_lora=true`): ships the tiny ~66 MB adapter via `/v1/load_lora_adapter` (vLLM-native hot-swap). Requires `--enable-lora` + the torch211 venv, **TP=1 only**. The IPEX BGMV PDE that historically blocked this is **already fixed** by the torch211 stack (no IPEX; validated 2026-05-05); the remaining cold-DAOS RPC hang is now mitigated in the launcher (node-local `/tmp` staging + 12-tile warmup gate + generous `VLLM_WARMUP_MAX_TIME`). Prefer `delta` over `runtime` — faster and TP-agnostic.
  - The launcher `run_qwen3_4b_lora_2node.sh` selects via `LORA_PUBLISH_MODE` (defaults to `merged` unless `LORA_USE_RUNTIME=1`→`runtime`), at the validated-safe G=8/max_gen=384 envelope (config's paper G=24/max_gen=512 is the documented banned:1 boundary). Failed publishes fail-fast (`lora.fail_on_publish_error`, default True). 4B `merged` baseline ~52-53s/step, 16.35 GiB; vs full-FT ~36s/16-seq, ~36.6 GiB (LoRA's win is memory + 6-7× cheaper backward, NOT step time — see `docs/reports/lora_vs_fullft_4b_parity_20260617.md`). `delta` targets the ~28s/step publish overhead specifically.
  - **Deliberately a standalone recipe, NOT a base subclass.** A subclass conversion was attempted and descoped (2026-06-17) — the fork shares zero byte-identical methods with the base and is a simpler single-mode (server-only, single-replicate) reimplementation; naive inheritance would take wrong branches / AttributeError. It is kept standalone and guarded against correctness drift: it references the shared `_maybe_unpermute_qk` (weight_sync) and `batch_level_advantages` (rewards), and binds the shared `vllm_http_generate` and `_setup_vllm_server_mode`. The drift guards live in `tests/torchtune/dev/rl/test_recipe_family_correctness_parity.py` (Q/K-unpermute + batch-advantages reachability, and that any standalone binder of `_setup_vllm_server_mode` sets `_dp_replicate`/`_is_shard_leader`).
- `recipes/dev/async_grpo_full_finetune_distributed.py` — **async GRPO** (XPU variant in active development)
  - Overlaps rollout generation with training backward pass via `RolloutProducer`
  - See `recipes/dev/async_grpo.md` and `experiments/async_grpo/` for current status

**Injected method pattern:** Functions in `torchtune/dev/rl/*.py` are bound to the recipe class via class-body assignment at class definition time (e.g. inside the class body: `_sync_weights_to_vllm = _weight_sync_module._sync_weights_to_vllm`). When you see `self._foo()` in the recipe and cannot find `def _foo` there, look for a `_foo = _<module>_module._foo` binding line and then in the `torchtune/dev/rl/` modules listed below.

### RL Infrastructure Modules (`torchtune/dev/rl/`)

- `torchtune/dev/rl/weight_sync.py` (~3790 lines) — all weight sync runtime logic:
  - `_sync_dedicated_vllm_weights()` / `_recv_weight_update()` — gloo broadcast for `dedicated_rank` mode (requires `_wsync_pg` and `vllm_param_iter()` — currently BioReason model API, override for other models)
  - `_sync_weights_to_vllm()` / `_sync_weights_to_vllm_xccl()` — XCCL broadcast for `server` mode
  - `_sync_weights_to_vllm_shm()` — shared-memory for `colocate_sleep` mode
  - `_run_vllm_generation_server()` — dedicated vLLM tile generation loop (BioReason-specific: calls `_embed_model.build_prompt_embeds()`)
  - `_setup_xccl_wsync_pg()` — creates `_xccl_wsync_pg` (separate from `_wsync_pg`)
- `torchtune/dev/rl/vllm_backend.py` (~1076 lines) — vLLM init and mode setup:
  - `_init_vllm_early()` — colocate mode early init (before CCL PG)
  - `_init_vllm_early_dedicated()` — dedicated rank early init (before CCL PG)
  - `_setup_dedicated_vllm_rank(cfg)` — generic vLLM rank setup: creates `_training_pg` (xccl, [0..N-2]) and `_wsync_pg` (gloo, [0, vllm_rank]), seeds gen params. Call `super()` then add model-specific setup.
  - `_setup_dedicated_training_pgs(cfg)` — mirrors the above on training ranks; **must call new_group in identical order to avoid deadlock**
  - `_generate_with_colocated_vllm()`, `_generate_with_dedicated_vllm()`, `_wake_up_vllm()`, `_sleep_vllm()`
- `torchtune/dev/rl/distributed.py` (~968 lines) — `_slice_trajectory()`, `_gather_trajectory()`, `device_empty_cache()`, `init_xpu_process_group()`
- `torchtune/dev/rl/loss.py` (~445 lines) — `GRPOSimpleLoss`, `GRPOLoss`, chunked forward/backward
- `torchtune/dev/rl/rewards.py` (~681 lines) — `math_reward_fn()`, `gene_recall_reward_fn()`, `format_reward_fn()`, `batch_level_advantages()`
- `torchtune/dev/rl/types.py` (~65 lines) — `GRPOTrajectory`, `GRPOStats` namedtuples
- `torchtune/dev/rl/async_rollout.py` — `RolloutProducer` for async generation/training overlap
- `torchtune/dev/rl/generation.py` — generation with logprobs
- `torchtune/dev/rl/packing.py` — sequence packing for batched training

### `_wsync_pg` vs `_xccl_wsync_pg`

These are two different weight sync process groups — do not confuse them:
- `_wsync_pg` — 2-rank gloo group `[0, vllm_dedicated_rank]` for `dedicated_rank` weight broadcasts. Created by `_setup_dedicated_vllm_rank` / `_setup_dedicated_training_pgs` in `vllm_backend.py`. Generic infrastructure, not BioReason-specific — BioReason was just the first user.
- `_xccl_wsync_pg` — XCCL group used for `server` mode XCCL weight broadcasts. Created by `_setup_xccl_wsync_pg()` in `weight_sync.py`.

### BioReason Modules (`torchtune/dev/bioreason/`)
- `torchtune/dev/bioreason/model.py` — `BioReasonModel`: ESM3 + GO encoder + Qwen3-4B backbone
  - `build_prompt_embeds(input_ids, protein_sequences)` → `[B, P, H]` CPU tensor
  - `build_full_embeds(prompt_embeds, completion_ids)` → `[B, P+C, H]` on device
  - `forward(inputs_embeds, attention_mask, position_ids)` → logits (takes embeddings, not token IDs)
  - `vllm_param_iter()` — yields `(hf_name, param)` for backbone params (used in weight sync)
  - `projector_state_dict()` — trainable projector weights for weight sync
- `torchtune/dev/bioreason/dataset.py` — `bioreason_rl_dataset`, `bioreason_collate_fn`
- `torchtune/dev/bioreason/reward.py` — `bioreason_reward_fn()` (GO-term F1 score)

### MoE / Expert Parallelism Modules
- `torchtune/modules/moe/experts.py` — `GroupedExperts` with BMM scatter-pad-gather (6.3× speedup over sequential)
- `torchtune/modules/moe/_parallelism.py` — EP AllToAll dispatch / combine
- Active work in `experiments/ep_parallelism/` and `experiments/qwen3_moe/`

### Configs and Launchers
- `recipes/configs/dev/production/` — all production configs (BioReason, Qwen3-30B-A3B, gene recall, etc.)
- `experiments/<topic>/` — all launchers; PBS `-o`/`-e` must point into the same subdir

### Other Reference Recipes
- `recipes/ppo_full_finetune_single_device.py` — PPO single-device (algorithm reference)
- `recipes/dev/grpo_full_finetune_distributed.py` — upstream sync GRPO (original upstream starting point)
- `recipes/full_dpo_distributed.py` — DPO distributed (FSDP pattern reference)

### Distributed Infrastructure
- `torchtune/training/_distributed.py` — FSDP sharding, `init_distributed()`, mesh setup
- `torchtune/training/` — checkpointing, metrics, activation techniques

## Current Status

The recipe hierarchy:

```
GRPOFullFinetuneDistributedXPU   (grpo_full_finetune_distributed_xpu.py)   ← base
└── GRPOBioReasonDistributedXPU  (grpo_bioreason_distributed_xpu.py)       ← subclass

LoRAGRPODistributedXPU           (lora_grpo_full_finetune_distributed_xpu.py)
                                  ← standalone (FTRecipeInterface), NOT a base subclass
```

**`docs/status.md` is the authoritative, dated record of current experiment state — consult it
first; the snapshot below is a coarse orientation and lags.** As of 2026-06-17:
- **AuroraGPT-2B GRPO** — most active area: XPU SFT recipes + multi-corpus mathmix → GRPO
  (success 2.8%→~6.5%, n=2), 8-node HSDP scale-up (~10× distinct-prompt throughput), and
  step-based checkpoint resume (resume past step budget / at a different lr). Capability ceiling
  ~8-9% GSM8K success — past that needs a capability lever, not more GRPO steps.
- **MoE / Expert Parallelism** — Qwen3-30B-A3B and Gemma4-26B-A4B; BMM expert kernel + EP
  AllGather/grad-release XCCL path validated (EP=8/16). **Effort paused** as of 2026-06-17
  (validated, not under active iteration).
- **BioReason multimodal GRPO** — ESM3+GO+Qwen3-4B; `dedicated_rank` and `server` modes
  validated; 2-node prototype + 200-step stability baseline.
- **Async GRPO** — `RolloutProducer` overlapping generation with backward; phase 1 validated
  on Qwen3-3B (not under active iteration).
- **Asym-optim** — `experiments/asym_optim/`; vLLM on rank subset (Phase A) + AdamW on spare
  XPU tiles (Phase B). Opt-in via `vllm_ranks` / `optimizer_offload.spare_ranks`. Phase A
  blocked by the colocate L0 urEventWait wedge (see status.md).

See `docs/status.md` for current experiment state and `memory/MEMORY.md` for investigation history.

**To add a new model recipe**: subclass `GRPOFullFinetuneDistributedXPU`, override `setup()` (call model-specific setup, then delegate RL params to base class helpers), override `generate_trajectory()` for model-specific embedding/forward, override `grpo_step()` if the forward signature differs. Use `grpo_bioreason_distributed_xpu.py` as the template.

## Experiment Organization

All launcher scripts and PBS outputs go in `experiments/<topic>/` — never in the repo root.
The set of topics grows over time (`ls experiments/` for the live list); the most active are:

```
experiments/
  auroragpt_2b_bakeoff/  # AGPT-2B GRPO production runs + cross-framework bake-off
  agpt2b_sft/            # AGPT-2B SFT pipeline (mathmix → GRPO handoff)
  lora_grpo/             # LoRA-GRPO + dense 4B server-mode runs
  ep_parallelism/        # Expert parallelism dev (EP AllToAll, Qwen3-30B) — paused
  qwen3_moe/             # Qwen3 MoE BMM kernel benchmarks
  bioreason/             # BioReason multimodal GRPO
  async_grpo/            # Async rollout overlap (RolloutProducer)
  asym_optim/            # vLLM-on-subset + spare-rank optimizer
  colocate/              # Fully-colocated (Ray TP=8, colocate_sleep)
  multinode_32b/         # 32B multi-node GRPO
  gemma4/  gene_recall/  wsync/  baselines/   # + others; not exhaustive
```

**Rule:** Set PBS `-o` / `-e` to the experiment subdir, not the repo root:
```bash
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/ep_parallelism/my_job.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/ep_parallelism/my_job.err
```

## Build & Test

```bash
module load frameworks
pip install -e /flare/ModCon/ngetty/torchtune       # Editable install
pytest tests/ --timeout=60                            # Always use --timeout
```

Tests requiring XPU must run on compute nodes, not login nodes.

### Run-health gate — check BEFORE trusting any runtime number (mandatory)

Runtime perf numbers can be silently corrupted by a degraded execution path that CPU
tests cannot see (e.g. the gloo-CPU-bounce `reduce_scatter` on a non-EP chunked run adds
~130s/backward; a varlen flag silently no-ops; a banned:1 mid-run). On 2026-06-17 a 4B
full-FT step time was reported as 274s (a 64-seq/fbs=2/ZeRO-3 → 32-chunk config artifact)
and a wrong "LoRA 5× faster" conclusion followed. Defenses:

```bash
scripts/check_run_health.sh <logfile>              # GREEN/DEGRADED verdict (exit!=0 on DEGRADED)
scripts/check_run_health.sh --compare logA logB    # A/B: same grpo_step path + transport?
scripts/check_run_health.sh --baseline 4b 36       # monotonicity vs known step-time baselines
```

**Rule (`docs/RESULTS_DISCIPLINE.md`):** no runtime number enters status.md / memory / a
conclusion without (a) a monotonicity sanity-bound vs a known baseline (a 4B can't be slower
than a 32B at the same topology), (b) `check_run_health → GREEN`, (c) for an A/B, both legs
verified same path/transport. When a fix doesn't move the number, the DIAGNOSIS was wrong —
re-diagnose from per-component timing, don't guess again. Gate it into launchers with
`scripts/check_run_health.sh "$LOG" || exit 1`.

### CPU-safe regression tests (`tests/torchtune/dev/rl/`)

Run on a login node — no XPU, no distributed init, fast (~7s):

```bash
module load frameworks
pytest tests/torchtune/dev/rl --ignore=tests/torchtune/dev/rl/workers --timeout=60 -v
```

The `workers/` subdir is excluded — those tests cover the Ray/CUDA stack, depend
on `tensordict`, and are not maintained for Aurora (see env-var section below).

What each file pins down (do not let these regress silently):

- `test_checkpoint_resume_state.py` — `setup()` must capture `OPT_KEY` and
  `DATALOADER_KEY` onto `self` BEFORE clearing `checkpoint_dict`. Resuming
  used to `NameError` because consumers ran after the cleanup.
- `test_device_empty_cache_xpu_noop.py` — `device_empty_cache` must NEVER call
  `torch.xpu.empty_cache` on XPU devices (FSDP + empty_cache leaks UR handles).
- `test_bioreason_path_discovery.py` — the BioReason model and dataset loaders
  must not contain the substring `glob.glob` (the stdlib glob hangs on DAOS/dfuse).
- `test_async_loss_combo.py` — parametrized over every YAML in
  `recipes/configs/dev/production/`. Enforces:
  - `async_generation.enabled=true` ⇒ loss is `GRPOLoss` AND
    `always_compute_rollout_logprobs=true`.
  - `async_generation.enabled=false` ⇒ `always_compute_rollout_logprobs=false`
    (anything else just wastes a policy fwd that `GRPOSimpleLoss` ignores).
- `test_rollout_versioning.py` — `_wait_for_sync_complete` bumps the version
  exactly once per dispatched sync (gated by `_pending_sync_id`). Without the
  guard the event starts in the set state and every call inflated the counter.
- `test_ep_slice_contract.py` — the recipe's pre-FSDP2 expert-weight slicing
  must match `ExpertParallel._token_dispatch`'s expert-ownership formula
  (`g = ep_rank + local_exp_idx * ep_degree`, i.e. interleaved). The pre-fix
  recipe used contiguous slicing; weights and tokens were silently permuted on
  every EP > 1 run. Test compares both formulas directly across several
  (num_experts, ep_degree) pairs and includes a sanity test that the broken
  contiguous formula must NOT match dispatch on any rank.

### Pytest plugins

The `pyproject.toml` `addopts` requires plugins not in the frameworks env:

```bash
pip install --user pytest-integration pytest-timeout
```

Without these, pytest aborts with `unrecognized arguments: --without-integration`
and conftest fails on `'Namespace' object has no attribute 'run_integration'`.

### Opt-in environment variables (recipe-level gates)

| Env var | Effect |
|---------|--------|
| `TORCHTUNE_USE_IPEX_VARLEN=1` | Routes causal-only SDPA through IPEX `varlen_attention` on **no-grad paths only** (ref fwd, rollout logprobs) — training forward always uses standard SDPA (varlen has no autograd kernel). XPU fast path; bit-exact. Requires `TORCHTUNE_MASKFREE_CAUSAL=1` to engage on dense Qwen3 (otherwise mask≠None and varlen no-ops). Kernel uses uniform seqlens (`[0, S, 2S, ...]`) — benefit is efficiency + no mask allocation + persistent output buffer. Validated: 19% faster on BioReason; 32–57% grpo_step faster on Qwen3-8B with maskfree. Grep logs for `varlen=engaged|requested-but-skipped|disabled|no-grad-only` to confirm. |
| `TORCHTUNE_MASKFREE_CAUSAL=1` | Suppresses explicit causal mask in `generate_trajectory()` so `mask=None` reaches attention layers, enabling IPEX varlen on dense Qwen3. Guards: XPU only, packing disabled, runtime padding check (falls back with warning for variable-length prompts). Validated Qwen3-8B: step 1+ = 32% faster wall-clock, grpo_step 45% faster, −2 GiB/tile at G=4; 57% faster, −4 GiB/tile at G=8. Speedup is in the policy forward (eliminates O(S²) mask allocation per layer). Prompt safety: `_generate_with_vllm` truncates to `vllm_max_model_len - max_generated_tokens`. See `tests/torchtune/dev/rl/test_varlen_maskfree_parity.py`. **NOTE:** On variable-length-prompt datasets (GSM8K, math, gene_recall), MASKFREE bails per-step and varlen no-ops — use `TORCHTUNE_VARLEN_NOGRAD_BYPASS=1` (next row) instead. |
| `TORCHTUNE_VARLEN_NOGRAD_BYPASS=1` | Drops the explicit causal+padding mask **only on no-grad ref+rollout forwards** in `grpo_step` (training fwd untouched). Engages IPEX varlen on variable-length-prompt datasets where `TORCHTUNE_MASKFREE_CAUSAL=1` would bail. XPU-only; requires `TORCHTUNE_USE_IPEX_VARLEN=1` to do anything useful. Validated 2026-06-10 dense Qwen3-4B 2N GSM8K: -3 to -7% total wall, -7 to -8% grpo_step, `ratios=1.0000` bit-exact, `approx_kl=0.000000` (training grads unaffected because training fwd keeps explicit mask). KL drift bounded — `kl_loss` term sees ~30× relative bump at converged-ref initialization (absolute 0.0007 → 0.02) but tracks baseline once kl_loss is in normal training range. Pad positions attend on no-grad paths; no cross-prompt leakage because cu_seqlens are per-row independent. Confirm engagement with `varlen=engaged` + "varlen no-grad bypass ENGAGED" log lines. See `memory/project_varlen_nograd_bypass_validated.md`. |
| `TORCHTUNE_USE_CHUNKED_LOSS=1` | **The name is historically inverted from what it does.** `=1` selects **single forward + single backward** over all `num_seqs` (one allgather/reduce-scatter pair per step — avoids per-chunk FSDP collective overhead on 32B). Default `=0` (or unset) uses **chunked** fwd+bwd at `forward_batch_size` (one allgather/reduce-scatter pair per chunk, with `set_requires_gradient_sync(False)` / `no_sync()` on all but the last chunk). For small models (e.g. AGPT-2B 2N) the chunked path is preferred — it bounds the per-backward L0-resource footprint and clears the sig#2/UR-handle wall faster. The `grpo_step()` entrypoint logs a one-shot `grpo_step path: SINGLE_BACKWARD\|CHUNKED_BACKWARD\|PACKED` line on rank 0 so the runtime choice is unambiguous. See `memory/feedback_torchtune_use_chunked_loss_is_inverted.md`. |
| `TORCHTUNE_PINNED_CPU_BUF=1` | Pinned CPU staging buffer for trajectory gather (~8.5× speedup, validated G=32). |
| `TORCHTUNE_VARLEN_CACHE_MAX=1` | FIFO cap on the IPEX-varlen no-grad output-buffer caches (`_varlen_out_cache`/`_alibi`/`_seqlens` in `attention_utils.py`). The caches reuse the output buffer per `(b,h,s,d,dtype,device)` for the varlen fast path; on a FIXED-seqlen workload only one key appears, but on VARIABLE-seqlen RL (GSM8K) `s` changes per step so the un-capped dict grew forever = a LIVE-memory leak in the no-grad ref forward. **DEFAULT CHANGED 8→1 (2026-06-25):** a leak census on the 4B-LoRA-2N server soak (job 8561648) measured the TRUE per-generation footprint at **~2.5 GiB** (≈36 distinct `(total_tokens, n_heads, head_dim)` bf16 buffers, ≈69 MiB each — NOT the one ~14.7 MiB buffer the 2026-06-24 cap was sized against). At cap=8 that is up to ~20 GiB of seqlen-keyed generations on the tile → `banned:1` PDE around **step 6** on a 64 GiB tile. cap=1 holds only the current generation (validated bit-flat 1.43→4.28→4.46→4.25→4.45→4.35→4.33 GiB steps 0-6, 8/8 clean, vs baseline staircase 3.97→7.01→9.46→12.11→14.67→CRASH). On variable-seqlen RL cross-step reuse is ~nil anyway; raise the cap for FIXED-seqlen (SFT) to recover consecutive-same-shape reuse. Off-path (varlen disabled) unaffected. See `docs/reports/colocate_pagefault_investigation_20260625.md`, `docs/reports/colocate_memory_varlen_and_chunked_vocab_20260624.md`. |
| `TORCHTUNE_LINEAR_LOSS_ALLOW_FSDP=1` | LoRA-GRPO only. Overrides the `LinearGRPOLoss` (chunked-vocab) requirement that the run be no-FSDP (`TORCHTUNE_COLOCATE_NO_FSDP=1`). `LinearGRPOLoss` projects `model.output` OUTSIDE `model.forward` (per seq-chunk, to avoid the full `[B,S,vocab]` FP32 logit tensor); under FSDP FULL_SHARD the output weight is resharded after forward, so the projection multiplies by a shard → wrong numerics + broken grads. Default (unset) fail-fasts unless no-FSDP. Set =1 ONLY under SHARD_GRAD_OP (params stay resident) — untested. See `docs/reports/colocate_memory_varlen_and_chunked_vocab_20260624.md`. |
| `TORCHTUNE_LORA_DELTA_BASE_CPU=1` | **vLLM-worker side** (set in the launcher's vLLM env). For `lora.publish_mode=delta` (merge-at-receiver): forces the cached base weights onto **CPU** instead of the resident XPU device. Default (unset) caches the ~6.77 GiB base on XPU so the per-step `B@A`+add merge runs on-device (~ms); a CPU merge was measured at ~20s/step for 4B (252 fp32 matmuls × 12 contending tiles). Set =1 only if a vLLM tile lacks ~6.77 GiB free HBM (accepts the slow CPU merge). See `torchtune/dev/vllm_weight_sync_worker.py:load_lora_base_from_raw`. |
| `TORCHTUNE_COLOCATE_PREFIX_CACHE=1` | **EXPERIMENTAL (default 0 for colocate).** Forces vLLM `enable_prefix_caching` back ON for `vllm_mode=colocate`/`colocate_sleep` (default is now OFF for colocate). In the colocate GRPO loop policy weights are re-published every step, so cross-step prefix reuse is semantically invalid (cached blocks computed under stale weights) AND a suspected per-step active-memory creep on XPU: `reset_prefix_cache()` clears the block table but the hashed prefix-block tensors stay referenced, growing the PyTorch caching-allocator floor ~0.5 GiB/step (job 8557588). Default-off flattens the floor. Set =1 only for A/B or a workload that genuinely benefits. See `torchtune/dev/rl/vllm_backend.py`, `memory/project_nofsdp_colocate_reclaim_20260623.md`. |
| `TORCHTUNE_COLOCATE_NO_FSDP=1` | **EXPERIMENTAL (default 0).** LoRA-GRPO `vllm_mode=colocate` only. Skips FSDP-wrapping the model (frozen ~8 GiB base stays full-replicated per tile — fits on a 64 GiB tile alongside in-process vLLM ~24 GiB) AND re-enables a REAL `torch.xpu.empty_cache()` between gen/train (`_colocate_reclaim`). Hypothesis under test: the UR-handle leak that bans `empty_cache` on XPU is **FSDP-specific** (`empty_cache` + FSDP storage `resize_()` leaks L0 UR handles); without FSDP units, reclamation may work cleanly like CUDA — the plain-resident TRL/A100 colocate loop, where reserved memory drops between phases instead of staircasing to `banned:1`. If validated, this is the simplest path to the paper `max_gen=1024` envelope at colocate. Adapter grads are manually all-reduced (already FSDP-ignored), so DP sync is unaffected; uses the cached-base merge (no summon). See `memory/project_colocate_warmup_at_max_20260623.md`. |
| `TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0` | **Default 1 (ON for colocate).** LoRA-GRPO `vllm_mode=colocate` only. Before the train loop, runs one throwaway vLLM generation at the max prompt+`max_generated_tokens` length (`ignore_eos`) AND one no-grad ref forward + one real `grpo_step` fwd/bwd on a synthetic `[num_seqs, vllm_max_model_len]` batch, so the peak vLLM-KV and FSDP activation/all-gather buffers all allocate at **step 0**. Fixes the colocate seq-length **staircase to `banned:1`**: XPU can't `empty_cache` (FSDP UR-handle guard), so when a mid-run rollout first exceeds all prior lengths both engines grab larger un-reclaimable buffers and `reserved` climbs until a GPU page-fault. Front-loading flattens the curve from step 0 → unblocks the paper `max_gen=384/512` envelope (validated-safe ceiling without it is `max_gen=128`). Runs on ALL ranks (FSDP collectives). Set `=0` to disable for A/B. No-op off colocate. See `docs/reports/lora_colocate_4b_20260618.md`, `memory/project_overnight_colocate_ur40_plan_20260618`. |
| `TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX=1` | **Default 0.** LoRA-GRPO `vllm_mode=colocate` only. Skips the `self._vllm_llm.llm_engine.reset_prefix_cache()` call in the per-step colocate weight publish. Added to A/B-isolate it as a suspect for the colocate generation page fault; **HW-verified NOT the trigger** (the run still faults 4/4 with it skipped, 2026-06-25), so default stays ON (the call fires). Keep =0 in production. See `docs/reports/colocate_pagefault_investigation_20260625.md`, `tests/torchtune/dev/rl/test_colocate_skip_reset_prefix.py`. |
| `TORCHTUNE_COLOCATE_QUIESCE_WSYNC=1` | **Default 0 — REFUTED mitigation, kept as a documented dead-end.** LoRA-GRPO `vllm_mode=colocate` only. Wraps the colocate `load_weights` publish in a `barrier()`+device-sync (before AND after) to remove any XCCL collective overlapping the in-process vLLM weight mutation. **Does NOT fix the colocate page fault** (A/B job 8559693: 6/6 crash, barrier confirmed engaged): a barrier only syncs rank *arrival*; the fault is caused by the trainer's XCCL/L0 context being *resident* during the publish, not by instantaneous concurrency (matches Ray TP=8 W10). Any serialization-based fix is futile — the fix must REMOVE the second L0 client (engine sleep/wake or server mode). Off-path (unset) is byte-identical to baseline. See `docs/reports/colocate_pagefault_investigation_20260625.md`. |
| `TORCHTUNE_COLOCATE_SLEEP_WSYNC=1` | **Default 0.** LoRA-GRPO `vllm_mode=colocate` only. Wraps the colocate `load_weights` publish in a vLLM KV-only sleep (`sleep(level=10)` — discards KV cache, weights stay resident so load_weights still writes) → publish → `wake_up()`. Targets the 2-factor colocate page fault by clearing vLLM's own KV/L0 paging state during the weight mutation (the barrier-based `QUIESCE_WSYNC` was refuted; this instead resets the engine's L0 paging). Enables `enable_sleep_mode` + the XPU sleep patch for the colocate engine at init (`vllm_backend.py`). Candidate fix under HW validation (A/B job; cell=sleep). Caveat: vLLM-XPU sleep has its own fragility (see `project_colocate_sleep_8b_validated`). Off-path (unset) byte-identical to baseline. See `docs/reports/colocate_pagefault_investigation_20260625.md`. |
| `TORCHTUNE_MEM_PROBE=1` | Enables the `mem_probe` diagnostic imports in `grpo_full_finetune_distributed_xpu.py` (default off; keeps logs clean and avoids the `experiments/multinode_32b/` path coupling). |
| `TORCHTUNE_ENABLE_RAY=1` | Unlocks `torchtune.dev.rl.workers` and the Ray async recipe registration. CUDA/NCCL only — do NOT set on Aurora. |
| `TORCHTUNE_EP_DEBUG=1` | Enables EP forensic prints in `torchtune/modules/moe/_parallelism.py` (NTPE-AG / EP-DISPATCH / EP-COMBINE / PRE-RS-BWD / `_ep_mem_probe`) and `experts.py` (`[v111-EXPERT-TOKENS]`, `[v161-EXPERT-EMPTY]`). Off by default — production EP runs stay quiet. SLOW-threshold prints in `_ep_all_gather` / `_ep_reduce_scatter` remain unconditional so real perf signals still surface. |
| `TORCHTUNE_EP_USE_XCCL=1` | Opt-in: routes `_ep_all_gather` and `_ep_reduce_scatter` through native XCCL on the EP process group (on-device) instead of the v151 gloo CPU-bounce. Default off because v141-v150 hit a deterministic OFI CQ deadlock at op #259 (RS-BWD) on this exact path. The 2026-05-01 slice fix does NOT change that XCCL+OFI interaction; verify with the EP correctness/smoke tests before any production envelope run. NTPE-AG histogram remains gloo regardless. Iter1 (2026-05-01): clean 3/3 with 794s vs 826s gloo baseline — modest 3.9% win on v10f production envelope. |
| `TORCHTUNE_EP_GRAD_RELEASE_XCCL=1` | Iter2 opt-in: routes `_ep_release_fsdp_unsharded_grads`'s per-FSDPParam all-reduce through native XCCL on the dp_shard PG instead of the gloo CPU-bounce (D2H → gloo → H2D → chunk). Same A/B safety pattern as `TORCHTUNE_EP_USE_XCCL`. Requires the recipe's `set_process_groups(...)` to register the XCCL dp_shard PG (already wired). Falls back to gloo when the XCCL handle is missing or the FSDPParam routes via the dp_replicate gloo PG. Iter2 (2026-05-01): clean 3/3 with **541s vs 794s iter1 (-32%) vs 826s gloo (-34.5%)** on v10f production envelope — dominant lever (per-chunk all_reduce 25s → 7.6s, 3.3×). Pair with `TORCHTUNE_EP_USE_XCCL=1` for stacked effect. See `docs/reports/qwen3_ep_grad_release_xccl_20260501.md`. |
| `TORCHTUNE_EP_WSYNC_LAYER_BATCH=1` | Default 0. Opt-in: in `_sync_weights_to_vllm_xccl`'s EP MoE path, batch the 3 expert projections of each layer (`gate_proj`, `up_proj`, `down_proj`) into a single `all_gather_into_tensor` on `_shard_pg` instead of three separate `all_gather` calls. Cuts wsync collective count from 3·L to L (Qwen3-30B-A3B EP=16: 144 → 48). Bit-exact equivalent — see `tests/torchtune/dev/rl/test_ep_wsync_layer_batch_equivalence.py`. WS6 measured 0% speedup (gather is bandwidth-bound, not latency-bound); off by default. Compose with `TORCHTUNE_EP_WSYNC_GATHER_ROOT=1` for orthogonal stacking. Off-path (env unset) is byte-identical to baseline. |
| `TORCHTUNE_EP_WSYNC_GATHER_ROOT=1` | Default 0. Opt-in: in `_sync_weights_to_vllm_xccl`'s EP MoE path, replace `all_gather` with `gather(dst=active_shard_rank=0)`. Only the active sender consumes the full expert tensor; the other `_ep_deg-1` ranks discard the AllGather output today. Gather sends `~(N-1)L` bytes into the root NIC per projection instead of fanning `N(N-1)L` bytes across the fabric. Bit-exact equivalent — see `tests/torchtune/dev/rl/test_ep_wsync_gather_root_equivalence.py`. Targets the WS5 wsync_gather=75s floor by hitting the realistic root-NIC bandwidth (~2.6 GB/s seen on broadcast on the same Slingshot fabric). Composes orthogonally with `TORCHTUNE_EP_WSYNC_LAYER_BATCH=1`. Off-path (env unset) is byte-identical to baseline. WS7 measured 5% regression (XCCL `gather()` likely implements as allgather+extract on the same fabric); off by default. |
| `TORCHTUNE_EP_WSYNC_SHARDED=1` | RESERVED (sender-side wire NOT yet implemented; receiver Phase A + sender CPU pin-down landed 2026-05-03). When enabled in the future, will skip `_shard_pg` AllGather entirely and have each trainer EP rank broadcast its **local** expert shard over a new per-rank cross-PG (`_xccl_wsync_sharded_pgs`); vLLM workers filter by `expert_map`. Receiver in `vllm_weight_sync_worker.py:_load_fused_moe_experts_sharded` is in place and triggers automatically when broadcast manifest entries carry `trainer_ep_rank`/`ep_degree` tags (no entries do today). Currently the flag emits a one-shot warning at sync time and falls through to the existing path. See `docs/reports/MoE_EP_status_ws8_ws10_design.md` §"WS10 Phase B+C — implementation specifics" and CPU pin-down `tests/torchtune/dev/rl/test_sharded_vllm_moe_sync_equivalence.py`. Off-path (env unset OR sender unimplemented) is byte-identical to baseline. |
| `TORCHTUNE_EP_WSYNC_FP8_WIRE=1` | Default 0. Opt-in: in `_sync_weights_to_vllm_xccl`'s EP MoE per-projection path, cast each rank's local expert shard to E4M3 with per-output-row scales, AllGather fp8 + scale, decompress on the active rank back to bf16. Cuts `_shard_pg` wire bytes ~2× (the WS5 ~70s wsync_gather floor at EP=16). NOT bit-exact — per-element err bounded by `row_amax/4` (E4M3 has 3 mantissa bits + bf16 recast). Mutually exclusive with `TORCHTUNE_EP_WSYNC_LAYER_BATCH=1` and `TORCHTUNE_EP_WSYNC_GATHER_ROOT=1` (logs and skips if either is set). Scope: ONLY expert `gate_proj`/`up_proj`/`down_proj`; norms/attention/router/embedding stay bf16. Do NOT enable on workloads sensitive to weight noise without a KL-drift smoke first. CPU pin-down: `tests/torchtune/dev/rl/test_ep_wsync_fp8_wire_equivalence.py`. Off-path (env unset) is byte-identical to baseline. |

**Table completeness is enforced.** `tests/torchtune/dev/rl/test_documented_env_flags_exist.py` fails if any flag in this table is absent from the code (catches "ghost" flags), and reports code flags missing from the table. The table above lists the **supported** gates. Additional `TORCHTUNE_*` flags exist in code but are intentionally NOT promoted here:
- **Debug-only / forensic** (do not use in production): the `TORCHTUNE_SKIP_*` family (`SKIP_GRPO_BACKWARD`, `SKIP_GRPO_STEP`, `SKIP_GRPO_UPDATE`, `SKIP_REF_FWD`), `TORCHTUNE_ASYM_MEMPROBE`, the `TORCHTUNE_RAY_COLOCATE_*` co-tenancy probes.
- **Transport/tuning knobs** with sensible defaults (document a row before relying on one): `TORCHTUNE_WSYNC_BACKEND`, `TORCHTUNE_WEIGHT_SYNC_PATH`, `TORCHTUNE_WSYNC_HTTP_TIMEOUT`, `TORCHTUNE_VLLM_FANOUT_MAX`, `TORCHTUNE_VLLM_BG_WSYNC` / `TORCHTUNE_BG_WSYNC_SEND`, `TORCHTUNE_VLLM_LARGE_AG_BYTES` / `_PREWARM`, `TORCHTUNE_VLLM_XPU_GLOO_TP`, `TORCHTUNE_XCCL_BATCHED_AG`, `TORCHTUNE_XCCL_HOST`, `TORCHTUNE_D2H_STREAM`, `TORCHTUNE_GRPO_BACKWARD_NO_SYNC`, `TORCHTUNE_EP_WSYNC_SHARDED_METHOD`.

### Async / weight-sync runtime guards

- `async_generation.enabled=true` with `max_staleness > 1` raises `ValueError`
  at recipe init. Behavior-policy logprobs are not captured at vLLM time yet;
  staleness=1 is the only regime where train≈behavior policy.
- A one-line warning is logged whenever async is enabled, naming the invariant.

## Code Style

- Standard recipes are **self-contained scripts** — copy-paste-modify, not subclass
- **Exception**: model-specific recipes subclass the general recipe. `GRPOBioReasonDistributedXPU` inherits from `GRPOFullFinetuneDistributedXPU` and overrides `setup()`, `generate_trajectory()`, `grpo_step()`, `train()`. Future model recipes follow the same pattern.
- Google-style docstrings, type annotations
- Device-agnostic: check `torch.xpu` → `torch.cuda` → CPU
- Logging: `logging.getLogger(__name__)` (no print in library code)

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Top-level FSDP only | Per-module wrapping causes catastrophic overhead on XPU |
| BF16 everywhere | XPU native precision; halves AllReduce communication |
| BioReason extracted to subclass | Model-specific logic (ESM3+GO, prompt_embeds, reward) doesn't belong in the general recipe |
| `dedicated_rank` PG setup is generic | `_wsync_pg` and `_training_pg` live in `vllm_backend.py`, not in model-specific code — BioReason was the first user but the infrastructure is general |
| Async GRPO uses IS-corrected loss | Off-policy trajectories from `RolloutProducer` require importance sampling; ratios ≈1.000× confirms on-policy regime at current overlap |
| MoE expert forward via BMM | scatter-pad-bmm-gather replaces 18K sequential kernel launches; 6.3× speedup on 6 XPU tiles |
