# K3 single-user: exact next steps (written 2026-08-11, end of session)

Everything here is committed and runnable. Both trees clean
(`vllm-xpu-src` @ `3ebc6dca`, torchtune @ `192dcfc0`).

## Where we are

| | |
|---|---|
| c=1 baseline | **1.041 tok/s** (961 ms/token), measured, 0.19% spread |
| with AR fusion (now default) | **1.138 tok/s** (879 ms/token), +9.3% measured |
| upstream reference | 118 tok/s @ c=1, GB300 TP16, no spec decode |
| ~~eager ceiling 7.2 tok/s~~ | **RETRACTED — see `DECODE_BUDGET_CORRECTED_20260812.md`** |

The "7.2 tok/s eager ceiling" was wrong in both factors: the step is 19,642
launches (not 21,293 — prefill was folded in) and in-situ per-launch cost is
~28 us (not 6.5 us — that microbenchmark looped a single op). There is no
wall. The measured budget of the 879 ms step is compute 130 ms (15%),
collectives 207 ms (24%), host/dispatch residual 542 ms (62%).

The conclusion that survives: the step is **dispatch-dominated**, so only
**fewer launches** (fusion) or **cheaper launches** (capture) matter. The
priority in section 2 below is REVERSED by the correction — capture's upper
bound is ~62%, KDA fusion's is ~17.5% (KDA is only 30% of launches).

## 0. State of the capture attempt as of end-of-session

Three attempts on hold 8748640. **No performance number was obtained.** But
the failures were distinct and all are now fixed:

| attempt | outcome |
|---|---|
| 1 | died in engine init: `logger.info_once` unpicklable under trace |
| 2 | died identically — my `@torch._dynamo.disable` fix was wrong (gb0098) |
| 3 | **zero errors**, workers init clean, weights 100% loaded, killed by walltime seconds before `Application startup complete` |

Attempt 3 proves the path is clean end to end through model construction:

    torch=2.11.0+xpu  supports_xpu_graph=True
    cudagraph_mode=PIECEWISE  enforce_eager=False   (0 errors)

So the remaining work is **only** to run it with enough walltime. Budget
**~25 min** for a cold-cache load (7.8 s/shard x 96) plus ~5 min to time.
A warm load is ~3 min, but do not count on warmth: the drain between legs
evicts the page cache, and attempt 3 was cold.

## 1. Finish the capture measurement (~20 min on a 3-node hold)

Blockers cleared this session: torch-2.11 venv prepared and verified;
`situ` wheel patch applied; `VLLM_XPU_ALLOW_GRAPH_WITH_COMMS` escape hatch
for `xpu.py:216`; `--enforce-eager` and `TORCH_COMPILE_DISABLE` made knobs;
`aiohttp_cors` installed; traced `logger.info_once` calls removed (they
killed all 32 workers under compile).

```bash
PYTHON=/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python \
RAY_ENV_MODE=torch211 REPEATS=2 \
LEGS="capture=ENFORCE_EAGER=0,CUDAGRAPH_MODE=FULL_DECODE_ONLY,VLLM_XPU_ENABLE_XPU_GRAPH=1,VLLM_XPU_ALLOW_GRAPH_WITH_COMMS=1" \
bash ab_c1_levers_3node.sh <FULL_PBS_JOB_ID>
```

**Pre-registered, REVISED 2026-08-12.** The old thresholds assumed dispatch
was 138 ms of 961; it is actually ~542 ms of 879. Against the current
AR-fused 879 ms default:

| dispatch removed | step | tok/s | delta |
|---|---:|---:|---:|
| all (upper bound) | 337 ms | 2.97 | +161% |
| half | 608 ms | 1.64 | +45% |
| quarter | 744 ms | 1.34 | +18% |
| PIECEWISE, ~93 segments each paying replay | — | — | plausibly single digits |

Decision: **>=+15%** pursue FULL capture (needs a capturable attention
backend); **+3-15%** real but PIECEWISE-bound — bank it and go to the fused
KDA kernel; **~0 or negative** replay overhead cancels the saving at 93
segments (consistent with Phase 0's 0.55-0.83x microbenchmark) — report and
stop, do not retry blindly.

Expect **PIECEWISE**, not FULL: `xpu.py:223-232` downgrades because
sycl-tla FMHA cannot be captured. Verify from `K3_WORKER_GATES` (each rank
echoes `torch=`, `supports_xpu_graph=`, `cudagraph_mode=`) before believing
any number.

## 2. Fused KDA decode kernel — worth having, NOT the biggest lever

**Corrected 2026-08-12.** KDA decode issues 85 kernel-launching aten ops per
layer x 69 layers = **5,865 launches, 30% of the step's 19,642** (counted by
`TorchDispatchMode` against the exact c=1 code path — run
`analysis/count_kda_decode_ops.py`). Fusing all of it to ~4 launches/layer is
worth **+17.5% upper bound**, charging removed launches at the residual rate.

The other 13,777 launches/step are MoE/MLA/norm/sampler, so the biggest lever
is whatever removes launches across all 93 layers at once — capture or
whole-graph fusion — not one hand-written kernel. Do this SECOND; it composes
with capture (capture removes per-launch cost, fusion removes the launches).

Upstream folds causal conv + recurrent update + RMSNorm into ONE launch.

Start from `kda.py::_kda_recurrent_xpu` (the vectorized decode branch) and
the chunked-prefill Triton work already in the tree. A Triton fused decode
kernel is the single largest available reduction.

## 3. Do NOT re-try (measured and refuted this session)

- collectives dominating the step — 20-28%, not 37-74%
- `ZE_ENABLE_API_TRACING=1` inflating launch cost — 6.58 vs 6.50 us
- host contention / Ray `--num-cpus=4` — 12 concurrent procs give 6.19 us
- byte-reduction on collectives — latency-bound (37x bytes -> 1.25x time)
- `@torch._dynamo.disable` around traced logging — gb0098, worse than the
  original failure

## 4. Traps that cost time here

- `PYTHON` and `RAY_ENV_MODE` must move together, else the head runs one
  torch and the 24 remote workers another. Now guarded in `serve_k3.sh`.
- Kineto traces from vLLM/Ray XPU workers have **no CPU rows and no
  collectives**; `analyze_decode_trace.py` refuses them rather than
  emitting a verdict from unattributed gap time.
- `delay_iterations>0` in `--profiler-config` captures NOTHING on this path
  (the step() hook never fires). Use 0.
- The loudest error is not always the real one: a `pybind11 ... not
  pickleable` RuntimeError was Ray failing to serialize the actual Dynamo
  exception.
