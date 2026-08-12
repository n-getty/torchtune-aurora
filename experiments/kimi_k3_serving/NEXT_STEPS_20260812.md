# K3 single-user: next steps (2026-08-12, supersedes NEXT_STEPS_20260811.md)

Both trees clean. torchtune @ `73ecb140`, vllm-xpu-src @ `bb489180`.

## Where we are — measured, not projected

| | tok/s | ms/token |
|---|---:|---:|
| 2026-08-11 baseline (torch 2.10) | 1.041 | 961 |
| + AR fusion (default ON) | 1.138 | 879 |
| + torch 2.11 venv | 1.154 | 867 |
| **+ fused KDA decode** (`VLLM_KIMI_XPU_KDA_FUSED_DECODE=1`, default OFF) | **1.267** | **789** |
| upstream reference | 118 | 8.5 |

Cumulative **+21.7%** from 1.041 over two sessions, all A/B'd in-allocation.

Budget of the 879 ms step: compute 130 (15%), collectives 207 (24%),
host/dispatch residual 542 (62%). See `DECODE_BUDGET_CORRECTED_20260812.md`,
but **apply a ~1.5x haircut to its launch-count projections** (below).

## 1. Fused KDA correctness prompt — DO THIS FIRST, it is cheap

The +9.8% is solid. The correctness evidence is not yet enough to flip the
default: the A/B smoke test prompts with 64 `a`s and the model answers 64
`a`s, which a numerically wrong kernel would likely survive.

Run both legs with a prompt that has a checkable answer and diff the
completions:

```bash
PROMPT='Q: What is 17 times 23? Think step by step.\nA:' \
MAX_TOKENS=64 REPEATS=1 GPU_MEM_UTIL=0.80 \
LEGS="base=;fused=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1" \
bash ab_c1_levers_3node.sh <FULL_JOB_ID>
```

(`ab_c1_levers_3node.sh` currently hardcodes the `a`-repeat prompt at line
~216 — parameterise it as part of this task.)

Identical, sensible text on both legs => flip the flag default-ON in
`serve_k3.sh` and add it to the launcher's standard set. Divergent text =>
the kernel is wrong on hardware despite 16 green CPU tests, which is itself
the finding.

## 2. Graph capture is BLOCKED — needs code, not a flag

Do not spend another hold trying to configure this. Measured 2026-08-12:
the capture leg loads all 96 shards, resolves
`cudagraph_mode=PIECEWISE enforce_eager=False`, then dies at

```
vllm/distributed/parallel_state.py:480
    assert isinstance(self.device_communicator, CudaCommunicator)
```

`graph_capture()` is CUDA-only by construction ("only cuda uses this
function"; calls `torch.cuda.current_stream()` directly). XPU has
`XpuCommunicator`, so it always fires. `VLLM_XPU_ALLOW_GRAPH_WITH_COMMS`
clears a *different, earlier* gate (`xpu.py:216`) and is not sufficient.

To pursue: add per-platform dispatch in `GroupCoordinator.graph_capture()`
(`torch.xpu.current_stream()`/`stream()`, skip the custom-allreduce context
that only `CudaCommunicator` has), or relax the assert to a minimal
interface `XpuCommunicator` can satisfy. Then the `xpu.py:223-232` FMHA
downgrade (sycl-tla kernels not capturable) is still a second limitation to
clear before FULL capture is reachable.

Upper bound remains ~49% of the step. Highest value, highest effort.

## 3. The other 13,777 launches

KDA was 4,140 of 19,642 launches and bought 77 ms. The rest are
MoE / MLA / norms / sampler. Same technique applies, but **calibrate with the
measured coefficient**: ~18.6 us per removed elementwise launch, not the 27.6
us the budget doc assumes.

Predicted vs actual for the KDA kernel: +13% predicted, +9.8% observed. Every
remaining launch-count projection is therefore ~1.5x optimistic. Either the
residual is non-uniform (KDA's ops are cheaper than the step-wide average) or
part of the 542 ms is fixed per-step cost fusion cannot reach — one more
fusion data point would separate those.

## 4. Sequence parallelism (collectives, 24%)

Untouched. Upstream's o_proj all_reduce -> reduce_scatter transformation.
AR fusion already took 92 of 463 collectives; this is the next structural
cut.

## Harness state — five fixes landed this session

All from real failures on job 8749119 (four failed model loads before two
good legs; only one cause was external):

- `GPU_MEM_UTIL` knob — 0.92 is not always attainable; one node had 52.6/64
  GiB free per tile with no user process on it. Use 0.80.
- `flock` per allocation — two concurrent drivers shared `RAY_TEMP_ROOT` and
  killed each other's workers.
- per-leg `RAY_TEMP_ROOT`.
- `ray stop --force` pre-flight — `pkill` strands Ray placement groups, which
  reserve every GPU and present as "Current node has no GPU available".
- collected-PID `wait` — a bare `wait` deadlocks against the `tee` from
  `exec > >(tee ...)`.

Operating rules learned the hard way: never edit a running bash script (copy
to /tmp first); don't SSH-probe compute nodes mid-run; `find -newermt
'-15 minutes'` silently matches nothing here, use `-mmin -15`.
See `memory/feedback_kill_vs_stop_and_wait_traps.md`.
