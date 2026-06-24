# BioReason-Pro RL throughput: CUDA/TRL baseline (Polaris A100) vs Aurora/torchtune (XPU)

> **Mirror copy.** The benchmark itself runs on **Polaris** at
> `/lus/eagle/projects/ModCon/ngetty/grpo_bench/` (that is the live, runnable location).
> This Aurora-repo copy is for reference. The harness + raw result JSONs are also mirrored at
> `experiments/bioreason/polaris_grpo_bench/` (`bench_grpo.py`, `run_polaris_grpo.sh`,
> `aggregate.py`, `results/*.json`). To re-run, do it on Polaris (see Reproduce below).

**Date:** 2026-06-23 · **Author:** ngetty (with Claude Code)
**Question:** We have no external-framework, CUDA-hardware GRPO baseline for the BioReason-Pro
RL workstream. What is TRL GRPOTrainer's step throughput on A100 at our prod envelope, so we
have a target to compare our torchtune-XPU numbers against?

## TL;DR

- **The upstream RL driver was never released.** `bowang-lab/BioReason-Pro` ships SFT only
  (PyTorch Lightning + unsloth) + a GRPO checkpoint *converter* + a model-side TRL plugin
  (reward/embed hooks). A literally-faithful "run their RL on CUDA" baseline is impossible.
- This is therefore a **text-proxy GRPO throughput benchmark**: TRL `GRPOTrainer` on Qwen3-4B
  at our prod envelope (G=8, max_gen=1024, LoRA r16/α32), measuring the *external framework's*
  GRPO loop cost. It matches prompt *length* (~2048 tok) but NOT the protein/GO modality.
- <!-- FILL: headline comparison once matrix lands -->

## Method

Single driver `bench_grpo.py` (companion to `sft_bench/bench_sft.py`): deterministic synthetic
prompts (seed 1234, ~2048 tok), length-based reward (non-zero group variance), warmup-discarded
median step time, generated-tokens/sec as the headline gen-dominated metric. Each result embeds
a config fingerprint + library versions + node id (self-auditing, per `docs/RESULTS_DISCIPLINE.md`).

**Envelope:** num_generations=8, max_completion_length=1024, per_device_train_batch_size=2,
num_iterations=1, steps_per_generation=1 (fresh rollouts every optimizer step → step time
includes generation), beta=0.04 (KL/reference fwd exercised), LoRA r16/α32/dropout0.05.

**Stack (Polaris trl-bench env):** trl 1.0.0 · transformers 4.57.6 · peft 0.19.1 ·
accelerate 1.13.0 · vllm 0.18.1 · torch 2.10.0+cu128. Model: Qwen3-4B (base; paper uses
Qwen3-4B-Thinking-2507 — parameter-identical for throughput, behaviour differs).

**Hardware:** Polaris node = 4× A100-SXM4-40GB (312 TFLOPS bf16). 1- and 2-node runs.
Both generation backends measured: HF-native `.generate()` and vLLM-colocate.

## Results (Qwen3-4B, G=8, max_gen=1024, prompt≈2414 tok, LoRA r16/α32, micro_bsz=2)

| run | nodes | gen backend | step median (s) | CoV | gen tok/s node | gen tok/s device | mean cmpl len | peak mem (GB) |
|-----|-------|-------------|-----------------|-----|----------------|------------------|---------------|---------------|
| 1N hf      | 1 | HF-native     | 75.4  | 5.6%  | 104 | 25.9  | 982 | 15.0 |
| 1N vllm    | 1 | vLLM-colocate | 12.6  | 10.2% | 594 | 148   | 961 | 33.2 |
| 2N hf      | 2 | HF-native     | 75.6  | 6.9%  | 208  | 26  | 988 | 15.0 |
| 2N vllm    | 2 | vLLM-colocate | 13.6  | 8.4%  | 1205 | 151 | 987 | 33.2 |

**Scaling 1N→2N:** step time is ~flat (vllm 12.6→13.6s, hf 75.4→75.6s) while gen tok/s/node
doubles (vllm 594→1205, hf 104→208). Correct GRPO behaviour — each rank generates its own group
in parallel, so adding nodes multiplies rollout throughput at constant step time (data-parallel
over prompts). The headline external-framework target is **vLLM-colocate ~12–14s/step,
~150 gen tok/s/device, ~600 (1N) → 1200 (2N) gen tok/s/node.**

### Why colocate "fits" on 40 GB A100 when Aurora colocate fails on 64 GB tiles — it is NOT a memory-fit problem
Evidence from Aurora colocate logs (`experiments/bioreason/colocate_v7_8549222_*.driver.log`,
2026-06-18) settles this — and corrects the premise that "we can't fit the model":
- **vLLM colocate DOES fit on the 64 GB tile.** Memory profile:
  `total 65520 MB, model load 7827 MB, free 56415 MB after profiling; GPU KV cache 54,016 tokens;
  init engine took 1.18s` (KV even manually capped via `num_gpu_blocks_override: 844`,
  `gpu_memory_utilization: 0.55`). vLLM + LoRA trainer coexist with ~45 GB to spare.
- **It trains for several steps:** `colocate LoRA W_eff sync 252 params in ~2.0–2.4s` ×5,
  `ratios=1.0000`, ~95s/step. Not an init-OOM.
- **It dies by `banned:1` GPU segfault / SIGABRT (exitcode -6)**, a GPU page-fault
  (`NotPresent ... PDP ... Write, banned:1`) after a few steps — the L0/UR-handle wedge class
  (`memory/feedback_banned1_destroys_xpu.md`, `bugs/project_xpu_emptycache_revalidated.md`),
  NOT out-of-memory.

**Conclusion:** the A100 completes the same colocate workload not because 40 GB beats 64 GB on
capacity (it doesn't), but because **CUDA has no equivalent leak/wedge — L0/UR handles are
reclaimed cleanly, so the run survives past the handful of steps where XPU gets `banned:1`.**
Capacity was never the binding constraint; an Intel L0 driver resource-accounting bug is. More
HBM cannot help a handle-count leak.

**Important correction:** sleep/wake is NOT the reason — this bench runs plain resident colocate
(`enable_sleep_mode` defaults False, not set), same as the torchtune path. The
`peak_mem_gb=33 GB` here is `torch.cuda.max_memory_allocated()` (training-side; excludes vLLM's
KV reservation). Caveats: this is TP=1 DP colocate (Aurora's other colocate failure, the TP=8
`urEventWait` wedge, is a distinct regime); and the text-proxy omits the resident ESM3 + GO
encoder + projection (several GB), so it understates the faithful multimodal footprint — but
that footprint difference is irrelevant to the conclusion, since the XPU failure is a leak, not
a byte-count ceiling.

**vLLM-colocate is ~6× faster per step and ~5.7× higher generation throughput than HF-native**
on the same A100 node — expected, and the reason our torchtune recipe (and the paper) use vLLM.
The vLLM rows are the apples-to-apples comparison to our vLLM-backed Aurora numbers; HF-native
is a lower bound. Generation dominates: mean completion length ~960–982 (near the 1024 cap),
matching the reasoning-length-bound regime we see on Aurora.

## Comparison to our Aurora/torchtune numbers (same G=8, max_gen=1024)

| Config (G=8, max_gen=1024) | Aurora/torchtune (XPU) | Polaris/TRL (A100) |
|----------------------------|------------------------|--------------------|
| LoRA 2N step, vLLM rollouts | ~26.7s/step (delta-publish) · ~52s (merged) | **13.6s/step** (2N vllm) |
| LoRA 1N step, vLLM rollouts | (not a standard XPU config) | **12.6s/step** (1N vllm) |
| Faithful 4N HSDP step | ~120–170s/step (gen-dominated, real multimodal) | — (proxy not run at 4N) |
| Gen throughput / device | — (not directly logged) | ~150 gen tok/s/device |

**Reading this honestly:**
- The Polaris LoRA-2N **13.6s/step is ~2× faster than our Aurora delta-publish 26.7s and ~4×
  faster than merged-publish 52s** — but the comparison is *not* clean: (a) the A100 run is a
  **text-proxy** (no protein/GO embed assembly, no resident ESM3+GO+projection), and (b) the
  Aurora step *includes weight-publish to a server-mode vLLM* (the delta/merged publish path),
  whereas the Polaris colocate engine reads weights in-process. The publish overhead is a real
  cost the proxy doesn't pay — so part of the 2× is mode (colocate vs server-publish), not raw
  hardware.
- The faithful Aurora ~120–170s/4N is **not** comparable to any proxy row — it carries the real
  2048-protein/200-GO multimodal pipeline + long reasoning traces. Its 3× over the old "45-53s"
  baseline is the faithful-envelope cost, documented separately
  (`docs/reports/bioreason_throughput_levers_20260623.md`), not a framework gap.
- The most defensible single takeaway: **on matched G=8/max_gen=1024 with vLLM rollouts, the
  external TRL framework on A100 sits at ~13s/step (LoRA, text-proxy). Our XPU LoRA server-mode
  is ~27s — the ~2× gap is split between (i) colocate-vs-server-publish mode and (ii) the
  XPU-vs-CUDA per-device efficiency + driver overhead already quantified in the SFT study
  (A100 ≈ 2.3-2.5× more efficient per device than a PVC tile).**

**Caveats (must read before citing):**
1. **Text-proxy, not multimodal.** No ESM3 protein embeds, no GO graph. Real BioReason gen has
   2048-protein + 200-GO prompt context; here only prompt *length* is matched. Our faithful
   step time is partly the multimodal prompt-assembly + embed-injection cost this proxy omits.
2. **Topology differs.** A100 node = 4 GPU (NVLink); PVC node = 12 tiles (XeLink). Our prod is
   2N/4N. Compare per-device gen tok/s and step time at matched G/max_gen, not node-vs-node raw.
3. **Generation backend matters.** HF-native decode ≪ vLLM throughput; our recipe is vLLM-backed,
   so the vLLM-colocate row is the apples-to-apples comparison; HF-native is a lower bound.
4. **Aurora node variance** ~1.8× job-to-job (`feedback_aurora_node_variance...`); only matched-
   config same-node A/Bs are strictly valid. These cross-cluster numbers are order-of-magnitude
   targets, not precision deltas.

## Reproduce

```bash
ssh polaris
cd /lus/eagle/projects/ModCon/ngetty/grpo_bench
# 1-node hf / vllm
qsub -q debug -l select=1:system=polaris -l walltime=00:30:00 -l filesystems=home:eagle \
  -v GEN=hf,G=8,MAXGEN=1024,STEPS=20,WARMUP=5,TAG=polaris_grpo_hf_1n run_polaris_grpo.sh
qsub -q debug -l select=1:system=polaris -l walltime=00:30:00 -l filesystems=home:eagle \
  -v GEN=vllm,G=8,MAXGEN=1024,STEPS=20,WARMUP=5,TAG=polaris_grpo_vllm_1n run_polaris_grpo.sh
# 2-node (select=2)
python aggregate.py   # collate results/*.json
```
