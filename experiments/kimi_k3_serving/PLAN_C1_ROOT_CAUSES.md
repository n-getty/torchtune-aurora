# Plan: attack c=1 at the root (2026-08-12)

Supersedes the retracted "20 tok/s is unreachable" claim in
`PATH_TO_20_TOK_S.md`. That document reasoned from a flop-bound intuition in
a bandwidth-bound regime and mistook our software's behaviour for the
hardware's limit.

## Corrected baseline and budget

Broken kernel OFF, so **1.154 tok/s = 867 ms/token**:

| term | ms | share | what it really is |
|---|---:|---:|---|
| compute | 130 | 15% | 0.9% of HBM peak — 12,405 elementwise kernels at ~5 us |
| collectives | 207 | 24% | 371 x 0.557 ms, 32-rank all_reduce |
| dispatch | **529** | **61%** | ~19,642 launches x ~27 us of host work |

Per-rank weight read is **0.93 GB = 1.16 ms** at 0.8 TB/s, and ranks run
concurrently. There is no hardware floor near 130 ms; we are simply running
12,405 tiny kernels where a few hundred large ones would do.

Targets: 20 tok/s = 50 ms (17.3x). 60 tok/s = 16.7 ms (52x).

## Ranked by (measured upside x confidence), highest first

### 1. Whole-graph compile — the one that attacks 61% of the step

**Status (2026-08-12, job 8750347): RUN. `compile_seg` = +20.5%**
(1.158 -> 1.395 tok/s), compile confirmed engaged (`mode=VLLM_COMPILE`,
0 graph breaks). Lands in the +5-25% "real, bank it, then explain" band;
dispatch term 527 -> 380 ms = 1.39x, short of the 3x assumed below.
`compile_whole` (`splitting_ops: []`, one graph) in flight.
**Correctness gate still pending** — both legs used the degenerate
`'a'` prompt. See `COMPILE_AB_RESULT_20260812.md`. The historical note
below is kept for the pre-registration it records.

**Status: verified reachable, never run.** This is the biggest miss of the
last two sessions: I queued it last, my own wrong ceiling justified
deprioritising it, and the flock refused it when the correctness rerun
overran.

Two facts checked in the source today:

- `gpu_worker.py:414-419` — `profile_cudagraph_memory()` (the call that hits
  the `CudaCommunicator` assert and blocks capture) is **skipped entirely
  when `cudagraph_mode == NONE`**. So compile-without-cudagraphs does not
  touch the blocked path.
- `compilation.py:1125` — `splitting_ops=[]` is rejected only under
  PIECEWISE/FULL_AND_PIECEWISE. At `cudagraph_mode=NONE` an empty list is
  legal, giving **one whole-model graph** instead of 93 attention-split
  segments.

Two variants, cheap to run back to back:

```bash
# 1a: default splitting (93 segments, intra-layer fusion only)
LEGS="compile_seg=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE"

# 1b: one graph, cross-layer fusion
LEGS="compile_whole=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE,VLLM_SPLITTING_OPS_EMPTY=1"
#     (needs -cc.splitting_ops='[]' plumbed through serve_k3.sh)
```

**Pre-registered.** 12,405 elementwise launches are ~133/layer. Inductor
typically fuses such chains 5-10x:

| elementwise fusion | launches | dispatch | step | tok/s |
|---|---:|---:|---:|---:|
| none (today) | 19,642 | 529 ms | 867 | 1.154 |
| 3x | 11,371 | 306 ms | 644 | 1.55 (+35%) |
| 10x | 8,401 | 226 ms | 564 | 1.77 (+54%) |

Decision: **>=+25%** -> this is the main line, iterate on it (autotune,
`custom_ops`, empty splitting_ops). **+5-25%** -> real, bank it, then look at
why fusion is limited (dynamic shapes? too many graph breaks?). **~0 or
negative** -> Inductor is not fusing on XPU; get the graph-break count from
`TORCH_LOGS=graph_breaks` before concluding anything.

**Risk:** XPU Inductor may fail to compile a 93-layer MoE graph, or take very
long to warm up. Budget a full leg and capture `torch._dynamo` logs.

### 2. Cut the elementwise count at the source (if 1 disappoints)

If Inductor cannot fuse them, the same 12,405 launches are still the target,
by hand. From the trace, per layer: ~42 + ~39 + ~33 + ~15 + ~5 = **~133
elementwise kernels/layer** in five families. Read the top offenders back to
source and fuse the obvious chains (norm+gate+scale, dtype-cast chains,
residual adds) into Triton kernels.

**Precondition:** any such kernel needs the hardware correctness gate from
day one — see the fused-KDA failure below.

### 3. PP=2 x TP=16 — real but modest, and it is a collectives play not a PP play

Checked today: **PP does not help c=1 latency.** With one request in flight
only one stage is active; the others idle. Each token still traverses all 93
layers. PP raises throughput via microbatching, which needs c>1.

What PP=2 buys is **TP=16 instead of TP=32**, which halves all_reduce width:

| | ranks | weights/rank | fits 68.7 GB? |
|---|---:|---:|---|
| TP=32 (today) | 32 | 48.8 GB | yes |
| TP=16 alone | 16 | 97.6 GB | **no** |
| **PP=2 x TP=16** | **32** | **48.8 GB** | **yes** |

So PP is the mechanism that makes TP=16 fit. Estimated 207 ms -> ~135 ms of
collectives = **~+9%**. Worth doing, not a route to 20 tok/s, and it costs a
config bring-up. Do it after 1.

### 4. Sequence parallelism / collective count (24%)

Untouched. Upstream's o_proj `all_reduce` -> `reduce_scatter` transformation.
AR fusion already removed 92 of 463. Composes with 3.

### 5. Fix or delete the fused KDA kernel

It is **numerically wrong on hardware** (corrupts recurrent state; see
`MORNING_HANDOFF_20260812.md`). 16 CPU-interpret tests missed it. Prime
suspects: in-place `tl.store` of the `[D,D]` state tile racing its own load,
the `tl.static_range` conv-window shift, or aliasing across programs sharing
a cache slot. Either debug on hardware with a small repro, or delete it —
carrying a fast-but-wrong kernel is a liability.

## What I will NOT claim again without measuring

- That any term is a "floor" before computing the hardware limit for that
  specific term (bandwidth for decode, not FLOPs).
- That a speedup is real before diffing generated text on a non-degenerate
  prompt.
- A ceiling derived from one microbenchmark. The 6.5 us/launch probe and the
  "flop-bound TP" intuition both produced confident, wrong conclusions.

## Order of work on the next hold (4 h)

1. `compile_seg` + `compile_whole` back to back (~2 legs, 70 min) — the 61%
   term. Verify from worker logs that `mode=VLLM_COMPILE` actually resolved.
2. Re-measure **c=128 clean** as the aggregate reference (~35 min); tonight's
   c=64 is not comparable to the 65.20 record.
3. Prompt-length ladder 16/24/32/41 at fixed chunking (~35 min) — the 41-token
   hang is a servability blocker and the cause is still unknown.

Items 3-5 above need their own hold.

---

## Pre-run verification (2026-08-12, before the compile legs)

Checked offline so a bad config could not silently produce a null result:

**Control reproduces across nodes.** `eager_base` on x4412/x4412/x4610 gave
**1.158 tok/s** (1.158 / 1.152 / 1.149, 0.8% spread, banned=0) against
1.152 median last night on x4320/x4407/x4608 — **+0.5%**. Node variance,
a documented Aurora confounder, is not affecting this A/B.

**Thresholds fixed against THIS control before seeing any compile number:**

| observed | reading |
|---|---|
| >= 1.447 (+25%) | compile is the main line, iterate |
| 1.216 - 1.447 (+5-25%) | real, bank it, investigate the limit |
| <= 1.193 (+/-3%) | no effect — **verify engagement before concluding** |

**The `SPLITTING_OPS_EMPTY` guard was tested both directions**, not just
assumed: with `CUDAGRAPH_MODE=PIECEWISE` it rejects (`requires
CUDAGRAPH_MODE=NONE`); with `NONE` it emits
`{"cudagraph_mode":"NONE","splitting_ops":[]}`. vLLM's own
`CompilationConfig` parses that to `cudagraph_mode=NONE splitting_ops=[]`.

**Engagement will be verified from the worker log, not assumed.**
`analysis/compile_diagnose.sh` reports the RESOLVED `CompilationMode`. A leg
that resolves to `NONE` is void as a compile measurement whatever its tok/s —
which is exactly how the capture attempt looked healthy right up until it hit
a second gate nobody knew about.

---

## RESULT: compile_seg = 1.395 tok/s, +20.5% — the first win on the dispatch term

Job 8750347, all legs one allocation, same nodes.

| leg | tok/s | reps | vs control |
|---|---:|---|---:|
| `eager_base` | 1.158 | 1.158 / 1.152 / 1.149 | — |
| **`compile_seg`** | **1.395** | 1.363 / 1.395 / 1.384 | **+20.5%** |

Non-overlapping (control max 1.158 < compile min 1.363), banned=0 both legs.
Step **864 -> 717 ms**, i.e. **147 ms saved = 28% of the 529 ms dispatch
term**.

**Engagement verified from the worker log, not assumed:**

```
mode = VLLM_COMPILE          cudagraph_mode = NONE
Dynamo bytecode transform: 64.4 s      graph breaks: 0      errors: 0
```

Zero graph breaks on a 93-layer MoE model — Dynamo traced it whole, so
Inductor had intact graphs to fuse. Compilation finished at 13:07:17, inside
the 30-min readiness window (deadline 13:15:19) with ~8 min to spare.

**This confirms the source reading**: `profile_cudagraph_memory()` is skipped
at `cudagraph_mode=NONE` (`gpu_worker.py:414-419`), so compile-without-
cudagraphs cleanly bypasses the `CudaCommunicator` assert that blocks capture.
The lever that sat untested for two sessions works.

### Against the pre-registered rules

+20.5% lands in the **"+5-25% = real, bank it, investigate the limit"** band,
just under the +25% that would have made it the main line. The launch-count
model predicted +36% at 3x fusion; we got 20.5%, so **the model over-predicts
here by ~1.75x** — consistent with the ~1.5x over-prediction seen on the
fused-KDA kernel. Two independent data points now say: treat launch-count
projections as roughly 1.5-1.8x optimistic.

### Why it is not larger, and what to try next

147 ms of 529 ms means Inductor fused a meaningful slice but far from all of
it. Likely limits, in order of cheapness to test:

1. **93 forced graph cuts.** `splitting_ops` defaults to the attention ops, so
   fusion cannot cross a layer boundary. That is exactly what `compile_whole`
   (`splitting_ops=[]`) tests — running now.
2. **`custom_ops`.** vLLM's default keeps several ops as opaque custom kernels
   Inductor cannot fuse through. Upstream's AMD recipe passes
   `custom_ops=["+fused_rms_norm_gated"]`; the inverse (`-all` to let Inductor
   own more of them) is worth an A/B.
3. **`mode=max-autotune`** on the hot GEMMs.
4. The remaining 382 ms may simply not be per-launch host work — the
   collectives term (207 ms) is inside it and compile cannot touch that.
