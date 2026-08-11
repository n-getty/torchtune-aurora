# The c=1 decode budget, corrected (2026-08-12)

This supersedes the launch-count arithmetic in `GAP_ANALYSIS_vs_UPSTREAM.md`
and in `memory/project_k3_eager_ceiling_and_first_win_20260811.md`. No new
hardware run was needed: both corrections come from re-reading the trace I
already had, plus a CPU-only op counter.

## Correction 1: the step is 19,642 launches, not 21,293

The 21,293 figure was `total trace events / 12 tokens`. But the 12 forward
passes are **1 prefill + 11 decode**, and prefill is nearly twice the size.

Segmenting the trace on a per-step marker kernel
(`ReduceKernel<unsigned int, long>`, which fires exactly once per forward and
appears exactly 12 times) gives:

| step | events | span ms |
|---|---:|---:|
| 0 (prefill) | 39,449 | 2510 |
| 1-11 (decode) | **19,642** each | 1460-1721 |

The decode count is **identical to the event on all 11 steps** — not a mean,
not a fit. `analysis/segment_decode_steps.py` reproduces this.

## Correction 2: per-launch cost is ~28 us in situ, not 6.5 us

The 6.5 us came from `probe_launch_contended.py`, which re-issues **one op**
(`x = x * 1.0001`) in a tight loop. That measures the async-submit floor of a
warm, single-op, no-allocation path. It is not what a real op costs.

Real median inter-kernel gap during decode: **44 us** (p25 33, p75 69).

**This kills the "7.2 tok/s eager ceiling" number.** That ceiling was
21,293 x 6.5 us = 138 ms. Both factors were wrong. The honest statement is
not a ceiling at all — it is a budget:

| term | ms | share | basis |
|---|---:|---:|---|
| device-busy (compute) | 130 | 15% | trace, union of kernel intervals |
| collectives (371 x 0.557) | 207 | 24% | measured 14 KiB 32-rank AR, isolated ⇒ **lower** bound |
| host/dispatch residual | 542 | 62% | remainder |

against the current default step of **879 ms** (post-AR-fusion). Nothing here
is unexplained — the earlier "43% unexplained" line was an artifact of
charging dispatch at the microbenchmark rate.

The direction of the old conclusion survives: **the step is dominated by
per-launch host work, not by compute or collectives.** What changes is that
the residual is ~28 us/launch of *real* work, so removing a launch is worth
~4x more than the old arithmetic implied, and there is no 7.2 tok/s wall.

## What this does to the fusion estimate

KDA decode issues **85 kernel-launching aten ops per layer** (148 total aten
ops, of which 63 are metadata-only views). Counted by `TorchDispatchMode` on
CPU against the exact code path taken at c=1, shapes from the K3 config at
TP=32 — `analysis/count_kda_decode_ops.py`.

    85 x 69 KDA layers = 5,865 launches = 30% of the step's 19,642

| stage | aten | launching |
|---|---:|---:|
| conv1d_update x3 (q,k,v) | 54 | 33 |
| gate (fused_kda_gate + clamp + sigmoid) | 17 | 11 |
| recurrence (`_kda_recurrent_xpu`) | 63 | 27 |
| o_norm (FusedRMSNormGated) | 14 | 14 |

Charging removed launches at the residual rate (27.6 us), **not** at the
whole-gap rate:

| fusion depth | launches cut | step | tok/s | delta |
|---|---:|---:|---:|---:|
| all four stages -> 4/layer | 5,589 | 725 ms | 1.380 | **+17.5%** |
| recurrence only -> 8/layer | 5,313 | 732 ms | 1.365 | +16.7% |
| everything but recurrence | 4,002 | 769 ms | 1.301 | +12.6% |

**Upper bounds.** They assume the residual is purely per-launch. Any fixed
per-step host cost does not shrink.

## The strategic consequence

KDA is 30% of launches. The other **13,777 launches/step are MoE, MLA, norms
and the sampler** — and 92 MoE layers at ~11 vllm::moe kernels each is only
~1,000 of them, so the bulk is generic elementwise in the dense path.

So a fused KDA kernel is worth having (+17.5% upper bound) but it is **not**
the "single largest lever" that `NEXT_STEPS_20260811.md` claimed. The largest
lever is whatever removes launches **across all 93 layers at once** — which
is graph capture, or `torch.compile`-style whole-graph fusion, not one
hand-written kernel.

That reverses yesterday's priority call. Capture's value was previously
computed as "+16.7% max, because dispatch is only 138 ms of 961". With
dispatch actually ~542 ms of 879, **capture's upper bound is ~62%, not 17%**.

## The two levers hit DISJOINT launch sets (this settles the priority)

`vllm::kda_attention` is registered via `direct_register_custom_op` and is a
member of `CompilationConfig._attention_ops` (`config/compilation.py:738-750`),
which is the **default `splitting_ops` list**. Under PIECEWISE the fx graph is
therefore cut at all 69 KDA call sites, and the op body is dispatched through
`forward_context.no_compile_layers` — it runs eagerly and is never traced or
captured.

**Consequence: capture and KDA fusion cannot reach the same launches.**

| | launches | residual ms |
|---|---:|---:|
| capturable (inside the graph) | 13,777 | 380 |
| not capturable (KDA, split out) | 5,865 | 162 |

| scenario | step | tok/s |
|---|---:|---:|
| now (AR-fused default) | 879 ms | 1.138 |
| + perfect capture of everything capturable | 499 ms | 2.00 |
| + KDA also fused to 4 launches/layer | 345 ms | 2.90 |

So capture's upper bound is **~43%**, not the ~62% the residual share alone
suggested — KDA's 162 ms is out of its reach by construction. And the levers
are **complementary**, not competing: each owns a disjoint share.

**The fused-RMSNorm anti-stacking result does not apply here.** That finding
(`memory/project_fused_rmsnorm_anti_stacks_with_compile_20260716.md`) was a
Triton kernel placed *inside* a compiled region, where it became an opaque
barrier and cost more Inductor fusion than it saved. `kda_attention` is
already a graph-splitting custom op running eagerly, so a Triton kernel there
displaces eager ops and blocks nothing. Worth re-checking empirically, but the
mechanism that caused the sign flip is absent.

## Revised priority

1. **Graph capture at TP=32** — ~43% upper bound, and attempt 3 already proved
   the path runs clean end-to-end. Needs only hold time.
2. **Fused KDA decode kernel** — ~17.5% upper bound, and it is the only lever
   that touches the 30% of launches capture structurally cannot. Stacks with
   1. Start CPU-side; no hold needed.
3. Collectives — 24% and already cut once by AR fusion. Sequence parallelism
   is the next structural cut, per upstream.

## Reproduce

```bash
module load frameworks
python3 analysis/segment_decode_steps.py <trace.json.gz>   # per-step counts
python3 analysis/count_kda_decode_ops.py                   # KDA op budget (CPU)
python3 analysis/decode_step_budget.py                     # the table above
```
