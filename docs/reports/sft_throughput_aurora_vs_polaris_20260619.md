# SFT throughput: Aurora vs Polaris (Qwen3-4B) — torchtune-native + corrected diagnosis

> ## ⚠ CORRECTION 2026-06-26 — READ FIRST (supersedes the step-time comparisons below)
>
> Two errors in the original report below are corrected here. The **t/s (throughput)
> columns below are valid**; the **step-time *comparisons* and the "Polaris 2.90×
> faster" conclusion are NOT**, and the Aurora step-time *values* are ~2× understated
> by node variance. Evidence: same-allocation A/B on hold 8563007 (node
> x4503c7s2b0n0), `memory/project_sft_compile_ac_qwen3_4b_20260625`.
>
> **Bug 1 — tokens/step are NOT identical across clusters (invalidates step-time
> comparison).** Aurora ran `world_size=12 → 49,152 tokens/step`; Polaris ran
> `world_size=4 → 16,384 tokens/step` (3× fewer; from the result JSONs
> `results/polaris/opt_polaris_full_fa2.json` ws=4 mb=2). So the line below claiming
> "tokens/optimizer-step = 49,152 identical → step times directly comparable" is
> **false**, and the **"Polaris 2.90× faster on full-FT"** (13.59s vs 1.56s step
> time) compares a 49,152-token step against a 16,384-token step — meaningless. The
> only valid cross-cluster axis is **throughput (tok/s/node)**.
>
> **Bug 2 — the Aurora 4.70s eager baseline is node variance (~2× understated).**
> Same-node remeasure: Qwen3-4B full-FT eager = **2.24s/step = 21,962 t/s/node** (not
> 4.70s / 9,934 t/s). `pin_memory` true vs false made zero difference (2.24 vs 2.24s),
> so it's not a config effect — it's the documented ~1.8× Aurora node-to-node spread
> (`feedback_aurora_node_variance_confounds_engine_ab_20260619`). The report's single
> Aurora run landed on a slow node.
>
> **Corrected cross-cluster picture (throughput, the valid axis):**
>
> | Full-FT | tok/s/node | vs Polaris full-FT node |
> |---|---:|---:|
> | Polaris (4× A100, TRL FA2) | 10,526 | 1.00× |
> | Aurora eager full-AC — report's slow node | 9,934 | 0.94× |
> | Aurora eager full-AC — same-node remeasure (8563007) | 21,962 | **2.09×** |
>
> Honest reading: Aurora eager full-FT throughput is **node-variance-bounded between
> ~0.94× and ~2.09× a Polaris node** — i.e. **roughly parity-to-ahead, NOT the "2.9×
> behind" the original step-time framing implied.** A clean claim needs several Aurora
> nodes (the original measured one). The "we never matched Polaris" premise was a
> measurement artifact, not a hardware deficit. (Aside: the ~12k often misremembered
> as the Polaris baseline is Polaris *LoRA* = 12,343 t/s; Polaris *full-FT* = 10,526.)
>
> **torch.compile is NOT a speedup on Qwen3-4B SFT on XPU (tested 2026-06-26).**
> Same-node: compile = 2.74s vs eager 2.24s (**0.82×, slower**); compile+selective-AC
> diverged the loss. Compile is now *unblocked* (`compile_dynamic=True` for the
> decoupled head_dim + scale_grads-compile-off on XPU — the `AssertionError: 2 != 1`
> was the compiled grad-scaler, not the model forward), but stays **OFF** for this
> workload. Inductor fusion doesn't beat eager SDPA+matmul on PVC here — matching this
> report's own "compile is a wash on compute-bound steps" prediction. Selective-AC
> every-2 is a marginal +9% (2.05s) at +13.5 GiB — situational. **Production optimum
> stays eager + full-AC** (fastest, lowest memory, correct). Details:
> `memory/project_sft_compile_ac_qwen3_4b_20260625`.
>
> ---

**Date:** 2026-06-19 · **Supersedes** the full-FT conclusion in
`sft_throughput_aurora_vs_polaris_handoff_20260619.md`.
**Hardware:** Aurora node = 6× PVC Max 1550 = 12 tiles · Polaris node = 4× A100-40GB.
**Harness:** `benchmarks/sft_throughput_aurora_vs_polaris/`.

All Aurora runs: Qwen3-4B, seqlen 2048, bf16, activation-ckpt ON, micro_bsz 2, grad_accum 1,
**tokens/optimizer-step = 12·2·2048 = 49,152** (⚠ see CORRECTION above: Polaris ran 16,384/step,
so cross-cluster step times are NOT comparable — use tok/s). torchtune runs use packed alpaca @
seq2048; TRL runs use a synthetic fixed-token dataset. Measurement = median per-step wall time
over 40–45 steps after 10 warmup.

---

## TL;DR

1. **torchtune-native FIXES the broken TRL full-FT cell: 13.59 s → 4.70 s/step (2.9×).** The TRL
   full-FT number in the handoff was a real artifact, but the cause was the **launch path**, not
   compute and not (as I first hypothesized) `CCL_WORKER_COUNT`. torchtune's `torchrun --standalone`
   gets the oneCCL XeLink fast path that TRL's `mpiexec`+accelerate FSDP could not.
2. **The engine delta cuts both ways.** On the *same* 12 tiles, torchtune is **2.89× faster than
   TRL for full-FT** but **0.55× (slower) than TRL for LoRA**. So a cross-cluster
   torchtune-Aurora-vs-TRL-Polaris number is confounded by engine — do not read it as pure hardware.
3. **The torchtune step is compute-bound at a ~4.2–4.7 s floor**, the same for full-FT, LoRA, and
   DDP-replicated LoRA. The bottleneck is the activation-checkpoint recompute + full-vocab loss, not
   the FSDP collective (which is XeLink-overlapped here). Confirmed by an AC-off probe (OOM at
   >50 GiB activations/tile).

---

## Results

### Headline — torchtune-native (the recipe we ship)
| Mode | Aurora (torchtune) | Polaris (TRL†) |
|---|---|---|
| LoRA | 4.44 s · 10,515 t/s | 1.33 s · 12,343 t/s · MFU 23.9% |
| Full-FSDP | 4.70 s · 9,934 t/s | 1.56 s · 10,496 t/s · MFU 20.3% |

† **Cross-engine caveat.** Polaris was NOT rerun on torchtune (its launcher is a draft; flare isn't
mounted on Polaris compute). Polaris numbers are TRL `SFTTrainer`, which on CUDA is healthy
(full-FT MFU 20.3%). Because torchtune and TRL differ in step time on identical hardware (next
table), the Aurora-vs-Polaris cells here mix engine with hardware. Treat them as suggestive, not a
clean platform ratio. The honest platform-only comparison is the same-engine TRL matrix below.

### Aurora engine delta — torchtune vs TRL on the same 12 PVC tiles
| Mode | torchtune | TRL | torchtune speedup |
|---|---|---|---|
| LoRA | 4.44 s | 2.46 s | **0.55×** (slower) |
| Full-FSDP | 4.70 s | 13.59 s | **2.89×** (faster) |

### TRL parity matrix (same engine both clusters — the original handoff comparison)
| Mode | Aurora (TRL) | Polaris (TRL) | Winner |
|---|---|---|---|
| LoRA | 2.46 s · 20,005 t/s · MFU 9.6% | 1.33 s · 12,343 t/s · MFU 23.9% | Aurora 1.62× node t/s |
| Full-FSDP | 13.59 s · MFU 1.7% ⚠ host-fallback | 1.56 s · MFU 20.3% | ~~Polaris 2.90×~~ ⚠ INVALID — see CORRECTION (diff tokens/step; use tok/s) |

### Single-device baseline (ws=1, mb=1)
| Device | step | tok/s | MFU |
|---|---|---|---|
| A100-40GB | 0.44 s | 4,619 | 35.7% |
| PVC tile (Max 1550) | 0.74 s | 2,752 | 15.8% |

---

## Corrected root cause of the broken TRL full-FT cell

The handoff narrowed the break to oneCCL host-fallback (`node_dev_uuids size 1` → no XeLink peer
group → ~1.75 GB/s host-staged all-gather). That is correct. Two refinements from this session:

- **`CCL_WORKER_COUNT=4` was NOT the bottleneck (falsified).** The TRL harness hardcoded `=4` (the
  repo's documented "48× AllGather regression" footgun). I hypothesized that explained the
  asymmetric full-vs-LoRA signature. Re-running TRL full-FT at `=1` left the step time **unchanged
  (~14.2 s)** and the topology warning still fired. Worker-thread count is irrelevant once the
  collective has fallen back to host staging. WORKER_COUNT=4 was a real misconfig (now `=1` for
  hygiene) but a red herring for this number. (Log: `logs/fix2_aurora_full.log`.)
- **The fix is the launch path, confirmed.** torchtune `torchrun --standalone` (all 12 device UUIDs
  visible per process, no per-rank `ZE_AFFINITY_MASK`) produced **zero** `node_dev_uuids size 1`
  warnings and ran the FSDP all-gather on XeLink → 4.70 s. This is the same fast path AGPT-2B SFT
  uses (67k tok/s/node). Details: `docs/bugs/accelerate_xccl_fsdp_topology_host_fallback.md`.

---

## Why torchtune LoRA isn't faster than full-FT (and DDP-LoRA only 5% faster)

torchtune's step is **compute-bound** at this envelope, so the levers that cut comm or weight-grads
barely move it:

| Variant | base layout | peak reserved/tile | median step |
|---|---|---|---|
| Full-FT | FULL_SHARD | 11.5 GiB | 4.70 s |
| LoRA | FULL_SHARD (base sharded 1/12) | 8.9 GiB | 4.44 s |
| LoRA DDP | replicate ×12 (base replicated) | 14.2 GiB | 4.22 s |

- **DDP-LoRA replicating the frozen base** (proven by the +5.3 GiB memory jump) removes the per-step
  base all-gather entirely, yet saves only **5%**. Because FSDP2 already overlaps the all-gather
  with compute, and on one node it's over XeLink — removing it recovers only the small exposed tail.
- **The ~4.2 s floor is the activation-checkpoint recompute + full-vocab `LinearCrossEntropyLoss`**
  (151,936 vocab). Disabling activation checkpointing OOMs at **>50 GiB activations/tile** (both
  full-FT and LoRA, at 61 GiB), proving activations dominate and the recompute is the real cost.
- Matches CLAUDE.md: *"LoRA's win is memory + cheaper backward, NOT step time."* TRL LoRA is faster
  (2.46 s) because PEFT keeps the frozen base replicated (no FSDP) — a different memory/throughput
  trade-off, not a free lunch (it can't full-shard a model that doesn't fit).

**`compile=True` is unavailable on this stack:** torch 2.10+xpu dynamo raises `AssertionError:
2 != 1` in `wrap_fake_exception` during graph capture (not an OOM). Eager is the validated optimum;
compile is documented as a wash on collective-bound steps and would not help a compute-bound one.

---

## Run-health / validity

- ⚠ INVALID (see CORRECTION at top): tokens/step are NOT identical — Aurora 49,152 vs Polaris
  16,384. The apples-to-apples comparator is **tok/s/node**, not step time.
- tokens/step identical (49,152) across engines → step time is the apples-to-apples comparator.
  tok/s differs by engine convention (TRL counts the full packed block; torchtune's per-gpu metric
  normalizes differently) — **trust the step-time column across engines**.
- torchtune full-FT: 45 measured steps, COV 0.37%, loss 2.16→1.06, zero topology warnings, native
  XCCL reduce_scatter (no gloo CPU-bounce). LoRA: COV 0.76%, loss 2.16→1.82. Both healthy.
- Monotonicity: pure SFT 4B full-FT at 4.70 s is well below the 4B *GRPO* floor (33–75 s, which
  adds rollout + ref-fwd) — passes the sanity bound. (`scripts/check_run_health.sh` reports
  DEGRADED on these logs only because it greps for GRPO-specific `TIMING step=` lines absent in SFT
  recipes — a false positive for this workload, not a real degradation.)

---

## Reproduce

```bash
# Aurora torchtune-native (held node, torchrun --standalone):
benchmarks/sft_throughput_aurora_vs_polaris/run_aurora_torchtune.sh   # MODE=full|lora
# configs: recipes/configs/dev/production/qwen3_4b_sft_throughput_aurora_xpu.yaml (+ _lora_)
# Aurora TRL (corrected, CCL_WORKER_COUNT=1):
qsub -v MODE=full,TAG=fix2_aurora_full benchmarks/.../run_aurora.sh
python3 benchmarks/sft_throughput_aurora_vs_polaris/aggregate.py   # rebuild all tables
```

## Open follow-ups
- **Polaris on torchtune** for a clean same-engine cross-cluster number (removes the engine
  confound). Needs a Polaris clone + eagle staging + venv (`setup_polaris_venv.sh` covers the TRL
  stack; torchtune needs the upstream device-agnostic recipe).
- **seq ≥ 8k sweep** — at seq2048 the step is MLP/loss-bound; longer context shifts the balance and
  would make FlashAttention-2 (Polaris) and any Aurora attention path actually discriminate.
