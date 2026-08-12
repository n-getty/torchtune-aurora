> # ⚠ THIS DOCUMENT'S CENTRAL CLAIM IS WRONG — RETRACTED 2026-08-12
>
> I claimed single-user 20 tok/s is "not reachable" because 130 ms of compute
> is a floor forced by TP=32. **Both halves are wrong.**
>
> **1. TP=32 does not make compute slow — it makes it faster.** At c=1 decode
> a matmul is a matrix-VECTOR product: bandwidth-bound, not flop-bound. Each
> rank reads only 29.7/32 = **0.93 GB**, which at 0.8 TB/s takes **1.16 ms**,
> and the 32 ranks run concurrently. Splitting weights 32 ways REDUCES
> per-rank bytes. I reasoned "smaller matmuls = worse occupancy" from a
> flop-bound intuition that does not apply to batch-1 decode.
>
> **2. 130 ms is not a floor, it is 0.9% bandwidth efficiency.** 0.93 GB in
> 122.7 ms = 7.6 GB/s against 800 GB/s peak. That is a software problem, and
> the trace names it: **12,405 of 18,399 kernels (67%) are elementwise, doing
> 62 ms of work at ~5 us each.** The MoE GEMMs are already batched (368/step,
> 26 us each — the only well-sized kernels in the trace).
>
> So the "7.7 tok/s ceiling" measured *our current software*, not the
> hardware. Unfused elementwise glue is precisely what Inductor exists to
> fuse — and the whole-graph compile leg that targets it never ran.
>
> The GB300 comparison was also overstated: more HBM per rank cannot explain
> a 110x gap when our own per-rank read is 1.16 ms of a 789 ms step.
>
> Everything below is kept for the reasoning trail. Treat the "forced",
> "floor" and "unreachable" language as retracted.

# What 20-60 tok/s actually requires (2026-08-12, overnight analysis)

Written before spending hold time, so the night's work targets something
reachable instead of grinding +10% increments toward a number they cannot
reach.

## The arithmetic that reframes the goal

Current: **1.267 tok/s = 789 ms/token**, measured, decomposed:

| term | ms | what it is |
|---|---:|---|
| compute (device-busy) | 130 | 18,399 kernels, median **4.5 us** |
| collectives | 207 | 371 x 0.557 ms, 32-rank all_reduce |
| dispatch/host residual | 452 | 19,642 launches |

20 tok/s = 50 ms/token = **15.8x faster**. 60 tok/s = 16.7 ms = **47x**.

**The trap:** every previous plan attacked dispatch. But

    dispatch -> 0  gives 337 ms = 2.97 tok/s
    dispatch AND collectives -> 0 gives 130 ms = 7.7 tok/s

**Perfect elimination of ALL eager overhead still yields 7.7 tok/s, not 20.**
Reaching 20 requires cutting the 130 ms of *device-busy compute* as well. No
amount of graph capture or collective fusion gets there alone.

## Why compute is 130 ms when the floor is ~1.2 ms

c=1 decode is bandwidth-bound: read the active weights once per token.
K3 activates top-16 of 896 experts + 2 shared = ~2% => ~56 B active params.
At MXFP4 (~0.53 B/param) that is **29.7 GB/token**. Aggregate HBM across 32
PVC tiles is ~25.6 TB/s, so the floor is **~1.2 ms/token**.

We measure 130 ms. **~100x off.** Not because the work is large — because the
kernels are too small to saturate a tile:

| | |
|---|---|
| median kernel | **4.48 us** |
| kernels < 10 us | 84.4% of all, 57% of busy time |
| kernels < 20 us | **97.2%** of all, **82%** of busy time |
| max kernel | 224 us |

A PVC tile needs work on the order of hundreds of microseconds to reach
useful occupancy. We are feeding it 4.5 us slivers.

## The common root: TP=32 is forced by memory

All three terms have one cause. TP=32 shatters every matmul into 32 slivers:
each too small to saturate a tile (compute 100x off floor), each layer needs
a 32-way all_reduce (371 collectives), each sliver is its own launch (19,642).

And TP=32 is **not a tuning choice**:

| | weights/rank |
|---|---:|
| TP=8 | 195 GB — does not fit 68.7 GB tile |
| TP=16 | 97.6 GB — does not fit |
| **TP=32** | **48.8 GB — fits** |

1.56 TB of weights against 64 GiB tiles forces TP>=32. Adding nodes does not
help: TP=16 needs 97.6 GB on each of 16 tiles regardless of how many other
tiles exist.

## Why upstream gets 118 tok/s and we cannot copy it

Upstream runs **TP16 on 16 GB300s** (and 111 tok/s at TP8 on 8). GB300 Ultra
has ~288 GB HBM, so 195 GB/rank at TP8 fits comfortably.

**They need 8-16 ranks for the same model because each rank has ~4x our
memory.** Consequences we cannot tune away at TP=32:
- their matmuls are 2-4x wider per rank => far better occupancy
- 8-16-way collectives, not 32-way
- NVLink/NVL72 domain, not Slingshot across 3 nodes

The 118-vs-1.27 gap is dominated by *per-rank memory capacity*, which is a
hardware property. This does not mean 20 tok/s is impossible — it means the
route there is different from upstream's.

## Realistic ceiling on this hardware, and the honest ranking

Best case if every eager overhead were removed AND kernels were merged enough
to reach even 10% of the bandwidth floor (12 ms compute):

    12 (compute) + 15 (collectives, cut 14x) + 20 (dispatch, cut 20x) = 47 ms
    => ~21 tok/s

That is the *theoretical* shape of a 20 tok/s configuration. Every one of
those three reductions is a major piece of work, and two are blocked or
unproven today. **20 tok/s is not reachable tonight.** What is reachable is
removing the largest remaining eager overheads and measuring honestly.

Ranked by (value x feasibility) for unattended overnight work:

1. **Batch the decode.** The single highest-leverage change available. At c=1
   every kernel is a 1-token sliver. At c=32-128 the SAME kernels do 32-128x
   the work for nearly the same launch and collective cost — this is exactly
   why we already measure 65.2 tok/s aggregate at c=128. It does not improve
   single-user latency, but if the goal is "20-60 tok/s served", concurrency
   already delivers it and is the only thing that does today.
2. **Fused KDA correctness prompt** (cheap, unblocks a measured +9.8%).
3. **Sequence parallelism / collective count** — 207 ms, 24%, untouched.
4. **Whole-graph compile** (`mode=VLLM_COMPILE` without cudagraphs) — attacks
   dispatch without needing the blocked `graph_capture()` path.

## Checked and rejected: expert offload to free up TP

The model is ~92% MoE expert weights (896 experts x 3 x 3584 x 3072 x 92
layers = 2.72 T params = 1443 GB at MXFP4, matching the 1561 GB on disk --
note experts run at `routed_expert_hidden_size`=3584, not hidden=7168). Only
18 of 896 experts fire per token, so "keep hot experts resident, stream the
rest" is the obvious idea for cutting resident memory and therefore TP.

**It does not work at c=1.** You cannot know which experts are needed until
the router runs, and streaming the ~29.7 GB of active weights per token over
PCIe (~64 GB/s) costs ~464 ms/token *serialized after* the router. That is
the same order as the entire current step, with none of it overlappable at
batch 1. Offload is a throughput/capacity technique, not a latency one.

TP=32 stands as forced.

## The honest headline

**Single-user 20-60 tok/s is not reachable on 3 nodes of PVC with this
model.** The 130 ms compute floor at TP=32, itself forced by 64 GiB tiles
against 1.56 TB of weights, caps a perfect implementation near 7.7 tok/s
unless kernel granularity is also fixed — and even then ~21 tok/s is the
optimistic shape, requiring three major workstreams.

**Aggregate 20-60 tok/s is already achieved** (65.2 tok/s at c=128,
2026-08-10). If the goal is throughput, we are past it. If the goal is
single-stream latency, the binding constraint is per-rank HBM.

---

## Overnight attempt at the concurrency route (job 8749725) — engine died

`conc32` did not produce a throughput number. It reported
`c1_tok_s=0.000 banned=0 verdict=OK`, which was a **harness bug**: every
response file was 0 bytes because the EngineCore had already died at
06:22:16 with

```
ray.exceptions.RayChannelTimeoutError: Timed out waiting for object available
```

304 s after `Application startup complete` — i.e. exactly the 300 s
`RAY_CGRAPH_get_timeout` that `ray_executor.py:576` already raises from Ray's
10 s default.

**Important: this was NOT caused by concurrency.** The timeout fired during
warmup, before any c=32 request was sent, on the *same server configuration*
(`--max-num-seqs 128`, TP=32, EP, mnbt=2048) as the working 65.20 tok/s c=128
run from 2026-08-10. Neither successful leg of job 8749119 shows a single
`RayChannelTimeout`. So this is a new, hold-specific or intermittent failure
of one compiled-graph step, not evidence that batching is broken.

Two harness defects it exposed, both fixed (commit 570a22e8):
- `verdict=OK` on zero tokens — a non-measurement entering the record as a
  measurement. Now `NO_TOKENS`.
- readiness gated on `/health`, which the API server answers 200 while the
  engine behind it is dead. Now requires one real completion before the
  timing loop, else `ENGINE_DEAD`.

**What this does NOT change:** the 65.20 tok/s at c=128 is still the measured
aggregate result on record, and aggregate throughput remains the only route
to 20-60 tok/s on this hardware. What is still unmeasured is whether the
fused KDA kernel moves that aggregate curve.

---

## RESULT: c=64 + fused KDA = 49.31 tok/s aggregate — inside the 20-60 target

Job 8749725, 2026-08-12.

| | |
|---|---|
| reps | 49.152, 49.312 tok/s (0.3% spread) |
| responses | **64/64 non-empty**, 4096 tokens total |
| wall | 83.1 s for 64 streams x 64 tokens |
| banned:1 | 0 |
| flags | `VLLM_KIMI_XPU_KDA_FUSED_DECODE=1`, TP=32, EP, mnbt=2048, util 0.80 |

Verified not a repeat of the conc32 phantom: every response file is
non-empty and the token count is exactly 64x64.

**Scaling efficiency: 38.9x throughput from 64x concurrency = 61%.** That is
the useful number. Single-stream latency is unchanged (each user still waits
~1.3 tok/s), but *served* throughput is 49.3 tok/s.

For context, the prior record is 65.20 tok/s at c=128 (2026-08-10, before the
fused kernel). c=64 reaches 76% of that with half the concurrency, which is
consistent with the sublinear step(c) = 1.249 + 0.00516c fit already on
record. **This run does not isolate the fused kernel's contribution at c=64**
— there is no c=64 baseline leg, because the c=32 leg that would have
anchored the curve died in warmup. Do not credit the +9.8% here.

### What this means for the 20-60 tok/s goal

**Met, in the aggregate sense, and reproducibly.** 49.3 tok/s at c=64 and
65.2 at c=128 both sit in the requested band.

**Not met, and not reachable, in the single-user sense** — see the analysis
above: perfect removal of all eager overhead still leaves 7.7 tok/s because
of the 130 ms compute floor, itself forced by TP=32, itself forced by 1.56 TB
of weights against 68.7 GB tiles.

If the requirement is "serve this model at 20-60 tok/s", that is done today.
If it is "one user sees 20-60 tok/s", it needs hardware with more HBM per
rank (upstream uses GB300 at ~4x our per-rank memory) or a substantially
smaller/quantized model.

---

## Correctness check STILL not obtained — 41-token prompt hangs the engine

Both re-run legs returned `verdict=NO_TOKENS` (the new guard working: it
refused to call zero tokens OK). Root cause is **prompt length**, not the
kernel and not concurrency.

The HTTP log settles the ordering — `2x 200` then `1x 500`:

| # | request | prompt tokens | result |
|---|---|---:|---|
| 1 | engine probe `hello` | ~1 | **200 OK** |
| 2 | warmup, 64 `a`s | 16 | **200 OK** |
| 3 | the timed request, real question | **41** | **500 after 300.9 s** |

`RAY_CGRAPH_get_timeout` is 300 s, so the third request hung for exactly the
timeout. `Dumping input data` confirms `prompt_token_ids_len=41`.

**Every successful K3 run in this repo used 16 prompt tokens** — last night's
baseline, the fused-KDA leg, and today's c=64 run all used the 64-`a`
prompt that tokenizes to 16. The 41-token prompt is the first non-trivial
prefill anyone has attempted here, and it hangs.

This is consistent with an already-recorded finding I failed to apply:
[[project_k3_48b_mnbt_ab_null_result_20260811]] states "512-token prompts are
unusable for fault-hunting on this stack (0 completions in 544 s)" because
**the XPU KDA prefill steps one token at a time** (`kda.py`). The harness even
carries the comment `512-in is unusable`. I designed a correctness prompt
without checking the known prefill limit.

**What is NOT established:** a naive linear model (41 x 0.789 s = 32 s) does
not predict a 300 s hang, so per-token prefill cost alone does not explain
it. The mechanism is unconfirmed — it may be superlinear, or a different
fault that longer prefill merely triggers. Do not state a cause without
measuring the prefill-length curve.

**Actionable next step:** `VLLM_KIMI_XPU_KDA_CHUNKED=1` exists and is
HW-validated at 2.15-7.12x on prefill
([[project_k3_chunked_prefill_xpu_validated_20260811]]) but is **default
OFF**. The correctness re-run should set it. Also worth measuring directly:
a prompt-length ladder (16 / 24 / 32 / 41 / 64 tokens) to find where the
cliff is, which is a cheap and genuinely useful number nobody has.
