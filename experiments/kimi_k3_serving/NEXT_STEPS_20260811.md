# K3 single-user: exact next steps (written 2026-08-11, end of session)

Everything here is committed and runnable. Both trees clean
(`vllm-xpu-src` @ `3ebc6dca`, torchtune @ `192dcfc0`).

## Where we are

| | |
|---|---|
| c=1 baseline | **1.041 tok/s** (961 ms/token), measured, 0.19% spread |
| with AR fusion (now default) | **1.138 tok/s** (879 ms/token), +9.3% measured |
| upstream reference | 118 tok/s @ c=1, GB300 TP16, no spec decode |
| **eager ceiling** | **7.2 tok/s** = 21,293 launches/token x 6.5 us |

The ceiling is the point. Collective removal, host-sync hoisting and kernel
tuning all operate *below* it. Only **fewer launches** (fusion) or **no
per-launch cost** (capture) change it.

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

**Pre-registered** (dispatch is 138 ms of 961, so this bounds it):
all dispatch removed = 1.215 tok/s (+16.7%); half = +7.7%; quarter = +3.7%.
Decision: >=+10% pursue FULL capture; +2-10% real but PIECEWISE-bound;
~0% replay overhead cancels the saving at this scale (matches Phase 0's
0.55-0.83x microbenchmark caveat) — report and stop, do not retry blindly.

Expect **PIECEWISE**, not FULL: `xpu.py:223-232` downgrades because
sycl-tla FMHA cannot be captured. Verify from `K3_WORKER_GATES` (each rank
echoes `torch=`, `supports_xpu_graph=`, `cudagraph_mode=`) before believing
any number.

## 2. Fused KDA decode kernel — the bigger lever (no hold needed to start)

Upstream folds causal conv + recurrent update + RMSNorm into ONE launch.
We currently spend **12,702 of 21,293 launches/token on elementwise ops**
doing only 62 ms of real work. This reduces the launch COUNT, so unlike
capture it is not bounded by `+16.7%`, and it does not depend on the
attention backend.

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
