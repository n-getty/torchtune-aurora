# BioReason 2-node: IPEX varlen_attention unblocks G=8

**Date**: 2026-04-30
**Hold**: 8460388 (debug, 2 nodes, 1h)
**Recipe**: `recipes/dev/grpo_bioreason_distributed_xpu.py`
**Config**: `recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml`

## Summary

Wired IPEX `intel_extension_for_pytorch.llm.functional.varlen_attention` as
an opt-in third branch in `torchtune/modules/attention_utils.py`, gated by
`TORCHTUNE_USE_IPEX_VARLEN=1`. Bit-exact vs PyTorch SDPA on XPU. **Pushes
the BioReason 2-node G=8 banned:1 ceiling from step 13 to ≥15** (full clean
exit). Production envelope can move from G=4 to G=8.

Side discovery: `force_math_sdpa` config flag is **a no-op on XPU** (toggles
`torch.backends.cuda.enable_flash_sdp` which doesn't reach the XPU SDPA
dispatcher). All "1-node uses math-only / 2-node uses optimized" framing
in prior reports was based on a phantom variable.

---

## Micro-benchmark (2026-04-30, single XPU tile)

Shape: B=8 S=1536 Hq=32 Hkv=8 D=128 (Qwen3-4B GQA, BioReason fbs=8 max_gen=1024).

| Backend                                | per call | 36-layer fwd | Peak MiB | Reserved Δ MiB |
|---------------------------------------|---------:|-------------:|---------:|---------------:|
| PyTorch SDPA (flash/mem-eff)          |     3.1  |       130.7  |   1057.5 |          788.0 |
| PyTorch SDPA (`force_math_sdpa=true`) |     3.2  |       131.6  |   1057.5 |          788.0 |
| IPEX varlen, fresh out per call       |     2.5  |       103.3  |    576.0 |          288.0 |
| **IPEX varlen, persistent out buffer**|   **2.5**|     **103.3**|  **384.0**|          **0.0** |

Findings:
1. `force_math_sdpa` toggle is **a no-op on XPU**. CUDA backend toggles
   (`enable_flash_sdp`, `enable_mem_efficient_sdp`) don't reach the XPU SDPA
   dispatcher. The 1-node config (no flag, defaults True) and 2-node config
   (explicit False) ran the same XPU SDPA path.
2. IPEX varlen is ~21% faster (3.1→2.5 ms per call) at BioReason shapes.
3. Persistent output buffer reduces per-call reserved delta to 0.
4. PyTorch's caching allocator gives 0 alloc events per call once warm in
   all backends — measurable `num_alloc_retries` is not the right
   per-call IPC handle proxy; cumulative L0 handle count from FSDP
   collectives is. The FSDP-step memory drop is the visible signal.

## Correctness validation

`/tmp/test_varlen_correctness.py`:
```
SDPA out norm:  109.80142974853516
varlen out norm: 109.80142974853516
max diff: 0.0  mean diff: 0.0
```
Bit-exact match across two seeds and shape configs. PyTorch's XPU SDPA
likely already dispatches to the same underlying IPEX kernel; our explicit
varlen path just exposes the persistent-output-buffer affordance.

## Live test (hold 8460388)

### Run A — sanity, G=4 NSTEPS=5 with varlen

| Step | total (varlen) | total (baseline) | grpo (varlen) | grpo (baseline) |
|------|---------------|------------------|---------------|-----------------|
| 1    | 45.0s         | 44.8s            | 14.1s         | 15.5s           |
| 2    | 51.5s         | 58.3s            | 10.8s         | 13.8s           |
| 3    | 51.7s         | 51.3s            | 11.5s         | 13.4s           |
| 4    | 53.0s         | 50.6s            | 11.7s         | 13.5s           |
| 5    | 50.9s         | 52.3s            | 11.3s         | 14.5s           |

- 5/5 clean, exit=0.
- Memory FLAT 52.87 GiB (matches non-varlen baseline).
- Avg grpo: **11.4s vs baseline 14.1s — 19% faster.**

### Run B — push G=8 NSTEPS=15 (baseline crashed at step 13)

| Step | total | grpo  | reserved (varlen) | reserved (baseline G=8) |
|------|-------|-------|-------------------|-------------------------|
| 1    | 52.2s | 17.6s | 47.36 GiB         | 49.46 GiB               |
| 2    | 58.1s | 14.2s | 54.89 GiB         | 59.44 GiB               |
| 5    | 57.3s | 14.9s | 57.21 GiB         | 62.28 GiB               |
| 8    | 56.7s | 14.8s | 57.21 GiB         | 62.28 GiB               |
| 12   | 57.1s | 15.5s | 57.21 GiB         | 62.28 GiB               |
| **13** | **62.6s** | **18.1s** | **57.21 GiB** | **62.28→51.44 → banned:1** |
| 14   | 54.7s | 14.0s | 57.21 GiB         | n/a                     |
| 15   | 55.9s | 15.0s | 57.21 GiB         | n/a                     |

- **15/15 clean, exit=0.**
- Memory FLAT at 57.21 GiB across steps 5-15 (vs baseline ramping 57→62
  then contracting at step 13).
- **5 GiB lower steady-state reserved** at G=8.
- Step 13 had a slight grpo time bump (15→18s) but no contraction — the
  previously fatal signature is now a mild transient that recovers.

## Production envelope update

Before: G=4 fbs=4 max_gen=1024 (banned:1 at G≥8).
After: **G=8 fbs=8 max_gen=1024** with `TORCHTUNE_USE_IPEX_VARLEN=1`.

At ~56s/step, that's **8 rollouts/step vs 4 = 2× rollout throughput**. Per
generated token throughput: G=8/56s × 1024 tok = 146 tok/s (sustained
across 15 steps), vs G=4/50s × 1024 tok = 82 tok/s previously.

G=12 not yet tested (hold time exhausted). Remaining unknown: whether
G=12/16 also survive or hit a new ceiling.

## Files changed

- `torchtune/modules/attention_utils.py` — added IPEX varlen branch in
  `_sdpa_or_flex_attention()` with persistent per-shape output/alibi/seqlens
  caches; gated by `TORCHTUNE_USE_IPEX_VARLEN=1`.
- `experiments/bioreason/run_bioreason_2node_server.sh` — added
  `TORCHTUNE_USE_IPEX_VARLEN` env var passthrough.
- `experiments/bioreason/varlen_test.sh` — sequenced A/B/C runner for
  hold-based testing.
- `/tmp/bench_attention_ipc.py` — micro-benchmark script (kept in /tmp;
  promote to `experiments/bioreason/` if needed for future bench rounds).
- `/tmp/test_varlen_correctness.py` — bit-exact correctness test.

## Memory updates

- `memory/project_bioreason_ipex_varlen_20260430.md` — new project entry.
- `memory/feedback_force_math_sdpa_xpu_noop.md` — new feedback (the
  config flag does nothing on XPU; future work should not vary it expecting
  behavioral change).
- `memory/MEMORY.md` — index entries for the above.

## Next steps

1. **Confirm G=12/16 ceiling** on next hold — does varlen survive arbitrary
   G or just push step ceiling out by a constant amount?
2. **Apply to other XPU recipes** that use SDPA: gene_recall (Qwen3-3B),
   Qwen3-30B-A3B MoE. Same wins likely apply (faster + lower transient mem).
3. **CLAUDE.md update**: drop or rewrite the math-only SDPA "precaution"
   guidance — it does nothing on XPU. If math-only is still desired as a
   precaution, it must be enforced by a different mechanism (e.g., setting
   `torch.backends.mha.set_fastpath_enabled(False)` or similar XPU-aware path).
4. **Reward-side flat-KL diagnosis still pending** — kl_loss stays ~0.003
   across runs. Independent of attention backend.
