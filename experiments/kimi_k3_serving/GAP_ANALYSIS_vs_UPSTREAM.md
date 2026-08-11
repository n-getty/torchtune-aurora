# Why we are at 1.04 tok/s and vLLM reports 118 tok/s at c=1

Written 2026-08-11, after the first direct c=1 measurement on Aurora and
after reading vLLM's Kimi-K3 blog + recipe.

## The two numbers are the same measurement

| | ours | vLLM upstream |
|---|---|---|
| c=1 tok/s | **1.041** | **118** (TP16), 111 (TP8) |
| step | 961 ms/token | 8.5 ms/token |
| hardware | 3 nodes x 12 PVC tiles, TP=32 | 16x GB300 NVL72, TP=16 |
| spec decode | none | none (370 tok/s is the DSpark number) |

**113x.** GB300-vs-PVC is worth maybe 3-5x. The remaining ~25x is
structural, and their own blog names the structures.

## The ceiling that makes this not a tuning problem

We issue **21,293 kernel launches per token** (measured from the trace) at
**6.5 us per launch** (measured on-node; unaffected by
`ZE_ENABLE_API_TRACING`, unaffected by running 12 processes per node).

    EAGER FLOOR = 21,293 x 6.5 us = 138 ms/token = 7.2 tok/s

That is with **zero** compute, **zero** collectives, zero everything else.
Our measured step is 961 ms, so every optimization I have been pursuing
lives in the 823 ms *above* a floor of 138 ms. Removing 100% of it yields
7.2 tok/s against a 20-70 tok/s target.

Conversely, 8.5 ms/token at our launch count would require 0.40 us/launch,
which is not achievable eagerly on any hardware. **Upstream is not
dispatching 21k kernels per token.**

So the plan's framing -- "cut collectives first, graph capture is
secondary" -- is backwards. Collective removal, host-sync hoisting and
kernel-level tuning all move us *within* a ceiling that only two things
change:

1. **Fewer launches** (kernel fusion)
2. **No per-launch cost** (graph capture/replay)

We currently have neither. `--enforce-eager` is hardcoded in
`serve_k3.sh`'s base ARGS array, and `xpu.py:216` refuses graph capture
whenever `world_size_across_dp > 1` (always true at TP>1).

## Their optimization list, mapped onto our measurements

| upstream optimization | our corresponding measurement |
|---|---|
| **fused KDA decode kernel** — folds causal conv + recurrent update + RMSNorm into ONE launch | our KDA decode is a Python-level sequence of elementwise ops; 12,702 of our 21,293 launches/token are elementwise |
| **KDA metadata builder**: 870 us -> 34 us at bs=1 (-96%), ~6% e2e | exactly the host-sync class my Step 3 hoist targets (~276 syncs/step -> 1) |
| **skinnyGEMM BF16** for skinny projections, ~10% e2e at small batch | our 2,826 GEMM launches/token do only 36 ms of work — latency-bound, same regime |
| **LatentMoE tail fusion** — reduce-scatter shared experts, ~7-8% e2e | this IS `VLLM_KIMI_FUSE_SHARED_EXPERT_AR` (implemented, under A/B now) |
| **sequence parallelism** — all_reduce after o_proj becomes reduce-scatter | our 463 collectives/step, 269 ms measured in-situ |
| **custom collectives 1.7-4.5x faster than NCCL** at small messages | our 14 KiB all_reduce = 0.557 ms, latency-bound (37x bytes -> 1.25x time) |
| **CUDA graph capture** + Rust frontend | we run `--enforce-eager`; XPU graph gated off at TP>1 |

Every single item we have independently measured as real. We were finding
the right problems and solving them in the wrong order, under a ceiling we
had not computed.

## Revised priority

1. **XPU graph capture at TP=32** — the only lever that changes the
   ceiling. Prerequisites already scoped: torch 2.11 venv (verified to
   exist, `situ` wheel patch mandatory, transformers 5.7.0 tested OK), plus
   an opt-in escape hatch around `xpu.py:216`'s blanket refusal. Phase 0
   already proved XPU graphs CAN capture and replay a real 2-rank XCCL
   all_reduce with correct numerics.
2. **Kernel fusion in KDA decode** — the fused conv+recurrent+RMSNorm
   kernel is the single biggest launch-count reduction available, and it is
   the same path our chunked-prefill work already touches.
3. Everything else (collective fusion, host syncs, skinny GEMM) — real,
   measured, but bounded by the ceiling until 1 or 2 lands.

## What the full upstream recipe adds (pulled from vllm-project/recipes)

Two findings that change what we do next:

1. **`VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1`** is in upstream's Blackwell
   env block. That is their name for the shared-expert reduce-scatter tail
   fusion — the same transformation as our
   `VLLM_KIMI_FUSE_SHARED_EXPERT_AR`, which I implemented from the algebra
   before reading their blog. Our vLLM checkout does NOT contain that flag
   (grep: no hits), so our implementation is not redundant — we are on a
   commit that predates it. Upstream credits it ~7-8% end-to-end, which is
   close to the ~5.8% I predicted for our leg from the measured collective
   term.

2. **The AMD profile runs cudagraphs**, and it is the closest precedent to
   XPU (non-CUDA backend, no FlashInfer):
   `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+fused_rms_norm_gated"]}'`
   `FULL_DECODE_ONLY` exists in our tree (`config/compilation.py:62`). This
   is a strong hint for the XPU path: capture decode only, leave prefill
   eager, which sidesteps the varying-shape problem that makes full capture
   hard. Note upstream's *disaggregated prefill* worker also runs
   `--enforce-eager` — eager is correct for prefill and wrong for decode,
   whereas we apply it to both.

Also worth noting: upstream's decode workers run `--max-num-seqs 32` and
`--max-num-batched-tokens 32` in the disaggregated profile, i.e. decode is
deliberately tiny-batch. Our `mnbt=2048` constraint (forced by the
`banned:1` fault above 2048) is therefore much less limiting for
single-user decode than it looked.

## Caveats

- 118 tok/s is TP16 on GB300 NVL72 with FP8 KV cache, FlashInfer MLA, and
  `--max-cudagraph-capture-size 256`. We have none of those, and some
  (FlashInfer) have no XPU equivalent.
- The recipe page is truncated; full profiles are in
  `github.com/vllm-project/recipes` -> `models/moonshotai/Kimi-K3.yaml`.
  Worth pulling before designing our own config.
- Closing to *parity* is not the claim. Closing 25x of structural gap to
  land in the 10-30 tok/s range is the plausible target, and that is within
  the original 20-70 ambition.
