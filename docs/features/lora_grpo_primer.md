# LoRA RL on Aurora — Primer

> **Status (2026-05-01):** Implementation complete. The recipe, helpers,
> config, launcher, and CPU-safe tests are all landed. See the "What we
> built" section below for the file inventory. The remaining step is a
> first end-to-end run on a held 2-node PBS job.
>
> Original document below is kept intact for context.

## Why this exists

The 200-step BioReason 4B full-FT stability proof
(`docs/reports/bioreason_4b_200step_stability_20260501.md`) validated the
engine — but reward stayed flat across all four quarters. Side-by-side with
the BioReason-Pro paper's own RL recipe, the gap is structural, not a knob:

| Knob              | Paper RL phase | Ours          | Gap                    |
|-------------------|----------------|---------------|------------------------|
| Trainable surface | LoRA r=16      | **Full 4B**   | architecturally diff.  |
| LR                | 3e-5           | 1e-6          | 30× too small          |
| KL coeff β        | 1e-4           | 1e-2          | 100× too aggressive    |
| Effective batch   | 192            | 11            | 17× too small          |
| Group size G      | 24             | 8             | 3× too small           |
| Temperature       | 1.0            | 0.8           | less exploration       |

Their hyperparameters are **co-designed with LoRA**: light adapters + tiny KL
"anchor the model to its base SFT behavior while exploring." Our full-FT path
can't just adopt those numbers. Even at 16 train nodes (288 tiles vs their 8
H100), our trajectory throughput stays ~3× short of the paper.

The full-FT engine work was the goal, and it's done. **LoRA-GRPO is a
separate, parallel goal**: a paper-comparable run on the same hardware in a
fraction of the capacity time, using the validated recipe.

## What we built (2026-05-01)

The full LoRA-GRPO stack is implemented. File inventory:

### New files

| File | Purpose |
|------|---------|
| `recipes/dev/lora_grpo_full_finetune_distributed_xpu.py` | Main recipe: `LoRAGRPODistributedXPU`. Server-mode only; FSDP1 `SHARD_GRAD_OP`; adapter-only optimizer; `disable_adapter()` ref path (no separate ref copy); `_publish_lora_to_vllm()` LoRA hot-swap. |
| `torchtune/dev/rl/lora_helpers.py` | Shared helpers: `build_qwen3_lora_model`, `adapter_optimizer_params`, `_translate_lora_key` (torchtune→PEFT name mapping), `torchtune_to_peft_state_dict`, `write_peft_adapter_dir` (atomic Lustre write), `load_lora_adapter_http`, `unload_lora_adapter_http`. |
| `recipes/configs/dev/production/qwen3_4b_lora_grpo_2node_server_xpu.yaml` | Paper-aligned config: LoRA r=16, lr=3e-5, KL β=1e-4, G=24, T=1.0, batch=192; 2-node server mode (11 train ranks + 12 vLLM HTTP tiles). |
| `experiments/lora_grpo/run_qwen3_4b_lora_2node.sh` | Launcher: node discovery, tile-mem precheck, `setsid+nohup` detached vLLM (with `--enable-lora --max-lora-rank 16`), cross-node health preflight, detached train + persistent watcher SSH. |
| `experiments/lora_grpo/batch_qwen3_4b_lora_2node.sh` | Self-terminating PBS wrapper (`qsub -v NSTEPS=200,...`). |
| `experiments/lora_grpo/probe_vllm_xpu_lora.py` | 50-LOC standalone smoke test: confirms `vllm_xpu_kernels._xpu_C` loads, `LLM(enable_lora=True)` constructs, `add_lora` / `remove_lora` API works. |
| `tests/torchtune/dev/rl/test_lora_name_translation.py` | CPU-safe: pins torchtune→PEFT key translation (all 7 modules, FSDP prefix strip, `output_proj→o_proj`, `adapter_config.json` fields). |
| `tests/torchtune/dev/rl/test_lora_adapter_params_complete.py` | CPU-safe: structural test using tiny `lora_qwen3()` (2 layers, embed_dim=64). Pins adapter param count, trainability, `adapter_optimizer_params()` flat list. |
| `tests/torchtune/dev/rl/test_disable_adapter_ref_path.py` | CPU-safe: `disable_adapter()` context manager — enters/exits cleanly, base vs adapter outputs differ (after non-zero lora_b init), no state leak across steps, bit-exact reproducibility. |

### Modified files

| File | Change |
|------|--------|
| `torchtune/dev/rl/vllm_backend.py` | Added `_lora_engine_kwargs(cfg)` helper; wired into `_init_vllm_early_dedicated()`, `_init_vllm_tp1()`, `_init_vllm_tp()` so any recipe with `vllm.enable_lora=true` in its config gets `enable_lora`, `max_lora_rank`, `max_loras` forwarded to the vLLM `LLM()` constructor. |

### Key design decisions made

| Question | Decision |
|----------|----------|
| vLLM LoRA-native vs merge-and-broadcast? | **LoRA-native**: vLLM hosts the adapter; rank 0 writes PEFT adapter dir to Lustre + POSTs `/v1/load_lora_adapter`. ~80 MB per sync vs 8 GB for backbone. |
| FSDP or not? | **FSDP1 `SHARD_GRAD_OP`**, top-level-only wrap (BioReason-validated topology). Adapter tensors are ~80 MB total so grad sharding is light; `use_orig_params=True` for cleaner state_dict. |
| Separate ref model? | **No**: `disable_adapter(model)` context manager is the ref path. Saves ~8 GiB HBM. Tested by `test_disable_adapter_ref_path.py`. |
| One recipe or subclass? | **Sibling recipe** (not subclass of full-FT base). The setup, optimizer, checkpointing, and wsync diverge enough that shared `train()` would be a liability; adapter-only training is a fundamentally different surface. |
| Lustre vs /dev/shm for adapter dir? | **Lustre** (`lora.shm_root`). `/dev/shm` is node-local; the 2-node topology needs both nodes to read the adapter dir. Named `shm_root` for config parity but must always point to Lustre in multi-node runs. |

## What we had before the implementation

- **Upstream LoRA infrastructure exists in this repo** (not Aurora-validated):
  - `torchtune/modules/peft/{lora.py, dora.py, _utils.py}` — `LoRALinear`,
    `AdapterModule`, `get_adapter_params`, `disable_adapter`, etc.
  - `torchtune/models/qwen3/_model_builders.py` — `lora_qwen3_4b_base` and
    `lora_qwen3_4b_instruct` are already defined component builders. Same for
    0.6B / 1.7B / 8B / 14B / 32B.
  - `recipes/lora_finetune_distributed.py` — multi-rank LoRA SFT (FSDP).
  - `recipes/lora_dpo_distributed.py` — multi-rank LoRA DPO (FSDP).
  - **No multi-node validation** of any of these on Aurora.
  - **No LoRA RL recipe** (no LoRA GRPO, no LoRA PPO) — this is what we needed.
- **Validated full-FT GRPO engine**:
  - `recipes/dev/grpo_full_finetune_distributed_xpu.py` — base recipe with
    train(), `_extract_batch_kwargs` hook, vLLM colocate/server/dedicated
    modes, FSDP1 setup, all the wsync paths.
  - `recipes/dev/grpo_bioreason_distributed_xpu.py` — multimodal subclass
    (ESM3 + GO + projector + Qwen3 backbone). 200-step stability proof.
  - `torchtune/dev/rl/{weight_sync.py, vllm_backend.py, loss.py, types.py}`
    — all weight-sync paths, vLLM init, GRPO loss, trajectory types.
  - `torchtune/modules/attention_utils.py` — IPEX varlen fast path
    (`TORCHTUNE_USE_IPEX_VARLEN=1`), validated bit-exact and 19% faster.

## What remains

1. **End-to-end smoke run** — `bash experiments/lora_grpo/run_qwen3_4b_lora_2node.sh`
   on a held 2-node job with `NSTEPS=3`. Requires Qwen3-4B weights at
   `/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B`. If weights are
   not present, download first (see config comment).
2. **Aurora-specific validation**: the LoRA paths have not been exercised
   on XPU multi-node yet. Expect the same class of issues as full FT —
   `device_id` in `init_process_group`, CCL transport, FSDP wrapping
   granularity, `empty_cache` UR leak — may surface on first run.
   The known-good env block in `CLAUDE.md` and the launcher decision
   table apply unchanged.
3. **BioReason LoRA variant** (not yet implemented) — subclass with ESM3
   frozen, GO encoder + projector trainable, backbone LoRA r=16. Same
   `_extract_batch_kwargs` hook pattern as `GRPOBioReasonDistributedXPU`.
   See original planning section below for design notes.

## The interesting design questions

These are the ones the planning session should answer.

1. **vLLM LoRA-native vs merge-and-broadcast?** vLLM supports LoRA
   adapters as first-class objects (per-request adapter selection). If
   the server hosts the adapter, our wsync becomes "broadcast 80 MB of
   adapter deltas every k steps" instead of "broadcast 8 GB backbone."
   Cost: vLLM API plumbing, may need version pin or workarounds for
   XPU. Alternative: merge LoRA into backbone on train side, ship
   merged backbone via existing wsync paths — works today but throws
   away most of the LoRA win.
2. **HSDP vs no FSDP at all?** With ~20M trainable adapter params,
   optimizer state is ~160 MB per replica. We could put the trainable
   side entirely on each rank (no sharding) and just AllReduce
   gradients across replicas. The frozen backbone is replicated
   per-rank anyway (vLLM-side too). This is a much simpler mesh than
   our current FSDP1 SHARD_GRAD_OP across 11 train ranks.
3. **What stays full-trainable?** Paper trains LoRA on backbone +
   keeps ESM3 frozen + trains projector. We need to make sure
   gradient flow reaches the multimodal projector through the LoRA
   adapters (not blocked by the frozen-backbone autograd setup).
4. **One recipe or two?** Subclassing the full-FT base recipe (like
   BioReason already does) keeps train() in one place — but a LoRA
   recipe diverges in `setup()`, optimizer, checkpointing, and
   wsync. Sibling recipe with shared helpers may be cleaner. The
   `_extract_batch_kwargs` hook precedent matters here: the more we
   keep `train()` itself shared, the more we benefit from the
   parity-test pin.
5. **Paper-aligned env first, then tune?** The paper's hyperparameters
   are validated together. Recommend running their exact numbers
   (LoRA r=16, lr=3e-5, β=1e-4, T=1.0, G=24, batch=192) first — if
   it works, we're done with hyperparameter exploration. If it
   doesn't, we have a known good reference to debug against.
6. **Scaling target?** Paper reaches batch=192 on 8× H100. With LoRA,
   our per-replica memory drops by ~50 GB (no full-FT optimizer
   state); 2-3 train nodes likely reach batch=192 without
   acrobatics. Target topology to plan for: 2 train + 1 vLLM
   (3 nodes), or 3 train + 1 vLLM (4 nodes) for paper-batch + headroom.

## What stays the same

The engine pieces that are validated and don't need rework:

- IPEX varlen attention path (assuming LoRA-wrapped layers route through
  the same SDPA call — needs verification).
- vLLM HTTP server-mode topology (12 servers per node, prompt_embeds
  wire for multimodal).
- XCCL `node_fanout` weight sync topology (drop-in for whatever new
  wsync we use; the topology env-var stays useful).
- The launcher hardening (`setsid` + `nohup` + persistent watcher
  SSH + tile precheck).
- The self-terminating PBS batch wrapper (`batch_bioreason_2node.sh`
  pattern is portable to a `batch_bioreason_lora_Nnode.sh`).
- All CCL env vars, the launcher decision table, the diagnostics
  (`BIOREASON_DIAG`, `METRICS`, `mem_probe` if enabled).
- Reward function (`bioreason_reward_fn` with the diagnostics).

## Out of scope for the planning session

- Switching the RL algorithm itself (GSPO instead of GRPO). Paper uses
  GSPO; we'd start with GRPO + LoRA to isolate the variable, and
  consider GSPO as a follow-up if convergence is slow.
- Re-validating full-FT EP results (separate workstream — see the EP
  slice contract fix in `tests/torchtune/dev/rl/test_ep_slice_contract.py`).
- Async GRPO with LoRA — keep the LoRA recipe synchronous first; A2
  async is paused on Aurora for unrelated bandwidth reasons.

## Concrete starting points

When the planning session opens fresh, the relevant files to read first:

- `recipes/lora_finetune_distributed.py` — upstream LoRA SFT pattern
  (FSDP wrapping, `get_adapter_params`, optimizer construction,
  `save_checkpoint`).
- `recipes/lora_dpo_distributed.py` — upstream LoRA RL-adjacent
  pattern (most relevant prior art for adapter-aware training loop).
- `recipes/dev/grpo_full_finetune_distributed_xpu.py` — our base
  GRPO recipe; `setup()`, `train()`, the wsync dispatch points, the
  `_extract_batch_kwargs` hook.
- `recipes/dev/grpo_bioreason_distributed_xpu.py` — how the
  multimodal subclass extends the base today; the LoRA bioreason
  recipe should follow the same shape.
- `torchtune/modules/peft/{lora.py, _utils.py}` — `LoRALinear`,
  `AdapterModule`, `get_adapter_params`, `disable_adapter`.
- `torchtune/models/qwen3/_model_builders.py:651` — `lora_qwen3_4b_base`
  signature and defaults.
- `recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml`
  — current full-FT config; the LoRA config will mirror its launcher
  and topology section.
- `CLAUDE.md` — Aurora platform constraints, CCL env vars, launcher
  decision table, opt-in env-var gates.

## Bottom line

The full-FT engine is built. LoRA-GRPO is a focused implementation task
that reuses ~80% of the engine, replaces the trainable surface, and
unlocks the paper's hyperparameters. Plan should answer the design
questions above (especially vLLM LoRA-native vs merge-and-broadcast)
before writing code.
