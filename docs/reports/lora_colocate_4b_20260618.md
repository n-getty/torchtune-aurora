# LoRA-GRPO per-rank colocate (4B) — implementation, UR:40 root cause, and step-time findings

**Date:** 2026-06-18
**Recipe:** `recipes/dev/lora_grpo_full_finetune_distributed_xpu.py` (`vllm_mode=colocate`)
**Configs:** `recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml`,
`recipes/configs/dev/smoke/qwen3_0_6b_lora_grpo_colocate_xpu.yaml`
**Launcher:** `experiments/lora_grpo/run_lora_colocate.sh`

## TL;DR

- **LoRA enables 4B colocate; full-FT does not.** Dense full-FT 4B OOMs at step-0 backward
  (full grad + AdamW co-resident with vLLM > 64 GiB tile). LoRA fits and runs.
- **4B LoRA colocate = 7.0 s/step** (on-XPU base merge, 30/30 clean) — or 11.0 s on the
  CPU-base fallback. **~2.5–4× faster than dense-server (~28 s/step)** at the matched
  AGPT 16-seq envelope.
- **Pure model-size ratio** (matched envelope, Qwen3 family, CPU-base): 0.6B 5.0 s → 4B 11.0 s
  = **6.7× params for 2.2× step time** (sub-linear). The original "4B looks 5–10× the 2B"
  was a confounded server-mode comparison, not a model-size effect.
- **UR:40 crash root cause:** vLLM KV / FSDP activation buffers grow as rollouts lengthen;
  XPU can't `empty_cache` (UR-handle-leak guard) so HBM staircases to `banned:1`. Bounded by
  `max_generated_tokens=128`. NOT a per-step leak, NOT the weight-sync.

## Measured table (4B, 2-node Aurora, 2026-06-18)

| config | step time | fits 4B? | notes |
|---|--:|:--:|---|
| 4B full-FT colocate | — | ❌ | OOM step-0 bwd (~48–50 GiB train-side) |
| 4B LoRA colocate, on-XPU base (default) | **7.0 s** | ✅ 30/30 | wsync 0.71–0.83 s |
| 4B LoRA colocate, CPU base (fallback) | 11.0 s | ✅ 30/30 | wsync ~5.4 s (CPU merge) |
| 4B dense server (non-LoRA baseline) | ~28 s | ✅ | gen ~6 s, grpo ~17 s, wsync_gather ~4 s (4 steps 27.7–28.8) |
| 0.6B LoRA colocate (ratio anchor) | 5.0 s | ✅ | wsync 1.27 s, gen 2.3 s, ref 1.3 s |

Throughput (4B LoRA colocate, on-XPU): 96 rollouts × 128 tok / 7 s ≈ **~1,750 gen tok/s/node**.

## Why LoRA+colocate beats dense-server (same 16-seq envelope)

- Generation: in-process per rank (~1.6 s) vs cross-node HTTP (vLLM 6 s).
- Backward: LoRA adapter-only vs dense's full-4B backward (grpo 17 s).
- Weight sync: in-process `load_weights` (sub-second on-XPU) vs cross-node XCCL gather (~4 s).

## UR:40 — the investigation (and three corrected diagnoses)

The 4B colocate run crashed with `banned:1` at step 8–22. Diagnosis took three wrong turns,
each corrected from data, not guessing:

1. **"Per-step FSDP summon leaks UR handles"** — disproved: caching the base once (no per-step
   summon) still crashed. (The cached-base design is still correct + landed; just not the cause.)
2. **"`max_num_seqs=64` over-provisions KV 8×"** — disproved: `vllm_backend._init_vllm_early`
   already computes `max_num_seqs = batch*grpo_samples` for colocate.
3. **"vLLM KV block pool grows"** — disproved: `num_gpu_blocks_override` (extended to plain
   colocate) didn't stop it.

**Actual cause (phase-probe instrumentation `COLOCATE_PHASEPROBE`):** the per-step `reserved`
growth is a **one-time, sequence-length-driven allocation** split between vLLM generation and
the FSDP `grpo_step` — when a rollout exceeds all prior lengths, both engines grab larger
buffers, and XPU can't reclaim them (`empty_cache` is a no-op by the UR-handle-leak guard).
It staircases until `banned:1`. **Fix:** bound `max_generated_tokens=128` so peak buffers fit
the step-0 budget → flat memory, 30/30 clean.

## On-XPU base cache use-after-free (found + fixed this session)

The cached-base merge has two HBM forms: CPU (slow ~5 s merge) and on-XPU (fast ~0.2 s).
On-XPU crashed at step 2 with a PML4 NotPresent-Read — **size-independent** (crashed at 4B
*and* 0.6B with 50 GiB free), so not OOM. Root cause: `_cache_colocate_base` snapshotted the
base *inside* `summon_full_params` with `.to(xpu).contiguous()`, which is a **no-op returning
the summoned (temporary) storage** for an already-XPU tensor — the cache aliased a buffer FSDP
frees on context exit → use-after-free. **Fix:** `.clone()` to force fresh storage. Validated:
30/30 clean on-XPU, wsync 5.4 s → 0.71–0.83 s, step 11 s → 7 s.

## Caveats / follow-ups

- On-XPU run's free HBM still gently declines (23 → 12 GiB over 30 steps) — survivable at 30
  steps, but a long run (>>30) or the paper `max_gen=384/512` envelope needs the **warmup-at-max**
  fix (front-load peak vLLM+FSDP buffers at step 0 so the staircase collapses to step 0). Not yet
  built.
- `max_gen=128` is the validated-safe colocate ceiling; the reward had a transient dip at
  step 11–12 (likely 128-token answer truncation on GSM8K) — a tuning matter, separate from
  stability.
- Server / dedicated modes are unaffected (vLLM not co-resident with the trainer).

## Artifacts

- Code: colocate path + `_sync_colocated_lora_weights` + `_cache_colocate_base` (recipe);
  `validate_vllm_mode` / `tune_lora_name_to_hf` (`torchtune/dev/rl/lora_helpers.py`);
  `num_gpu_blocks_override` extended to plain colocate (`torchtune/dev/rl/vllm_backend.py`);
  in-place merge `delta.add_(base_w)` (`lora_helpers.iter_merged_lora_layers`).
- Tests (CPU, all green): `tests/torchtune/dev/rl/test_lora_colocate_{gate,name_translation,merge_equivalence}.py`.
- Memory: `project_overnight_colocate_ur40_plan_20260618`, `project_lora_colocate_implementation_20260618`.
- Env flags (diagnostic): `TORCHTUNE_COLOCATE_MEM_PROBE`, `TORCHTUNE_COLOCATE_CACHED_BASE` (default 1),
  `TORCHTUNE_COLOCATE_BASE_CPU` (default 0 = on-XPU).
