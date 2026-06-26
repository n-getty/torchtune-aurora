# BioReason native-Gemma4 SFT — implementation report + smoke runbook (2026-06-26)

Implements the scoping plan `docs/plans/bioreason_32b_sft_scoping_20260626.md`: a BioReason
multimodal SFT recipe on the **native torchtune Gemma 4 31B** decoder.

## What was built (all CPU-validated, 13/13 tests green)

| Piece | File |
|-------|------|
| `lora_gemma4` / `lora_gemma4_31b` builders (r=32/α=64, handles global `k_eq_v` Identity v_proj, output tied) | `torchtune/models/gemma4/_component_builders.py`, `_model_builders.py`, `__init__.py` |
| `BioReasonNativeModel` — wraps `gemma4_31b()`/`lora_gemma4_31b()`, reuses `tok_embeddings` (sqrt scale), grad-enabled splice, freezes LoRA base, LoRA-merge-for-save | `torchtune/dev/bioreason/model_native.py` |
| `BioReasonSFTDataset` + collate — reserved-id placeholder splice (262142/262143), prompt-masked labels, reasoning+final_answer target | `torchtune/dev/bioreason/dataset_sft.py` |
| SFT recipe — subclass of `FullFinetuneRecipeDistributedXPU`; overrides `_setup_model`/`_setup_data`/`_loss_step`/`save_checkpoint` | `recipes/dev/sft_bioreason_distributed_xpu.py` |
| Configs — production + smoke | `recipes/configs/dev/production/sft_bioreason_gemma4_31B_xpu.yaml`, `recipes/configs/dev/smoke/sft_bioreason_gemma4_31B_smoke_xpu.yaml` |
| CPU tests A–H (+merge) | `tests/torchtune/dev/bioreason/test_native_gemma4_sft.py` |

CPU tests pin: model builds + reuses tok_embeddings; splice positions; grad flow through
splice; placeholder-id contract; **embedding sqrt-scale equivalence (Test E, the #1 risk)**;
count fail-fast; LoRA builder + base-freeze + bf16 adapters + tied output; LoRA-merge math;
recipe uses grad-enabled splice; config/dataset placeholder-id agreement; prompt byte-parity
with the RL dataset.

Run: `module load frameworks && pytest tests/torchtune/dev/bioreason/test_native_gemma4_sft.py --timeout=120`

## Pre-existing failures (NOT introduced here — verified by stashing my diff)
- `test_bioreason_lora_peft.py` (5): environmental `peft` 0.19.1 / torchao version-detect
  ImportError in the **HF-PEFT RL path** (untouched).
- `test_reward_target_and_propagation::test_dataset_source_has_no_go_pred_default` (1): naive
  substring match against `dataset.py`'s go_pred prompt-injection (untouched).

## HW smoke RESULT (2026-06-26, node x4108c2s5b0n0, job 8570128) — GREEN

1-node / 6-tile smoke, `sft_bioreason_gemma4_31B_smoke_xpu.yaml`, validation shard:
- **31B LoRA loads + FSDP2-shards at 9.66 GiB/tile** (10.26 GiB reserved) — fits a 64 GiB
  tile with huge headroom; could scale seq/batch or drop to fewer tiles.
- **5/5 SFT steps at seq=8192**, loss 41.9 → 35.4 (clear learning), ~25–30 s/step, setup ~25 s.
- Four bugs the smoke surfaced and fixed (committed): meta-device `.to()` on projections;
  `_SideInputDataLoader.__len__`; **FSDP2 DTensor embed-splice must run inside `forward`**
  (was outside → `aten.embedding` mixed Tensor/DTensor); seq budget — the BioReason prompt
  is large (p99 4618, max 5046 tok) so truncate the TARGET only, never the prompt; raised
  `max_seq_len` 4096→8192 (Gemma4 native max).
- Per-step time will inform the full-run node-day estimate (sanity-bound: SFT no-gen ≈ 25–30 s
  vs GRPO ~117 s with gen — consistent).

Remaining for a production run: scale to 2–4 nodes (full train shards, ~22K steps), confirm
the checkpoint save round-trips through the gemma4 vLLM overlay, then F_max eval.

## HW smoke runbook (Task #6 — needs a held node)

PREREQUISITE — build the ESM3 cache over the SFT sequences (ESM3 needs XPU, do on a tile):
```bash
python experiments/bioreason/precompute_esm3_cache.py \
  --data_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/data \
  --out /lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/esm3_cache_2048.pt \
  --max_protein_len 2048
```
(The smoke config points `esm3_cache_path` here; the validation shard's 7365 seqs must be covered.)

1. **CPU gate** (login node): tests above green.
2. **1-node smoke** (hold node + SSH; debug queue): launch the recipe with the smoke config
   (`sft_bioreason_gemma4_31B_smoke_xpu.yaml`, 5 steps, seq=4096, FSDP2, AC). Confirm: GEMMA4
   weights load into `backbone.*` (strict-ish; only adapter keys init-from-scratch), finite
   loss, backward + optimizer step, **memory fit on 6 tiles**. `scripts/check_run_health.sh "$LOG"`.
   Watch `Traceback|Error|OOM|banned|elapsed`.
3. **Measure `time_per_step_s`** → node-day estimate; sanity-bound vs GRPO ~117s/step (SFT no-gen
   must be lower). No number to status.md without GREEN health.
4. **2–4 node scale**, confirm no_sync/ZeRO-2 reproduces, no banned:1, short run.
5. **Eval** (Task #6 remaining): the save override writes a MERGED Gemma4 HF checkpoint + the
   projections. The GEMMA4 `FullModelHFCheckpointer` runs `gemma4_tune_to_hf` and writes HF
   safetensors. CAVEAT (eval-branch work, verify post-smoke): `eval_cafa_fmax.py` builds the embed
   layer via `BioReasonModel` (HF `AutoModelForCausalLM`) and serves with vLLM + the gemma4 overlay
   (`recipes/dev/vllm_gemma4_overlay/`). The merged HF checkpoint is a custom `gemma4` arch — the
   eval embed-layer load + vLLM serve must use the overlay, NOT stock AutoModel. Expect a small
   eval adapter (load `tok_embeddings` from the saved safetensors via the gemma4 key map; serve the
   merged dir under the overlay). **Success = F_max 0.66–0.70.**

## Open risks for the smoke
- 31B + multimodal + seq=4K memory on 6 PVC tiles — may need more nodes / activation offload /
  smaller effective batch. Measured at step 2.
- vLLM Gemma4 overlay must load the merged HF safetensors for eval (the overlay exists:
  `recipes/dev/vllm_gemma4_overlay/`). Confirm the merged checkpoint round-trips.
- The merged-backbone save assumes constant α/r across layers (true for the config).
