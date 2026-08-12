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
