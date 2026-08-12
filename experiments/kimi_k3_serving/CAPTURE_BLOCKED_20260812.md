# XPU graph capture at TP=32 is blocked in vLLM's distributed layer

**Job 8749119, 2026-08-12. Definitive negative result — stop pursuing capture
via env flags.**

## Result

| leg | tok/s | verdict |
|---|---|---|
| `baseline_t211` (eager, torch 2.11) | **1.154** (1.144 / 1.154 / 1.152, 0.9% spread, banned=0) | OK |
| `capture` (PIECEWISE, enforce_eager=False) | — | **NO_START** |

The capture leg is not a flaky failure. It loaded **all 96 shards**, resolved
the gates exactly as predicted:

    torch=2.11.0+xpu  supports_xpu_graph=True
    cudagraph_mode=PIECEWISE  enforce_eager=False

and then died in `determine_available_memory` ->
`profile_cudagraph_memory()` -> `graph_capture()`:

```
File "vllm/distributed/parallel_state.py", line 480, in graph_capture
    assert isinstance(self.device_communicator, CudaCommunicator)
AssertionError
```

## Why this is structural, not a config problem

There are **two** gates, and I only knew about the first:

1. `xpu.py:216` — the blanket "XPU Graph doesn't support capture
   communication ops" refusal. This is the one
   `VLLM_XPU_ALLOW_GRAPH_WITH_COMMS=1` bypasses. **Bypassing it works** — the
   run got past it and resolved to PIECEWISE.
2. `parallel_state.py:480` — `graph_capture()` asserts the device
   communicator is a `CudaCommunicator`. The surrounding comment says it
   outright: *"only cuda uses this function, so we don't abstract it into the
   base class"*. The body then calls `torch.cuda.current_stream()` and
   `torch.cuda.stream(stream)` directly.

On XPU the communicator is `XpuCommunicator`, so the assert always fires. No
environment variable can move this: **any** capture path at TP>1 routes
through `graph_capture()`, which is CUDA-only by construction.

## What it would take

Making capture work at TP=32 on XPU requires a code change in vLLM's
distributed layer, not a flag:

- teach `GroupCoordinator.graph_capture()` to dispatch per-platform
  (`torch.xpu.current_stream()` / `torch.xpu.stream()`, and skip the
  custom-allreduce context that only `CudaCommunicator` has), **or**
- give `XpuCommunicator` whatever minimal surface `graph_capture()` needs and
  relax the assert to that interface.

That is a real upstream-shaped change with its own correctness burden. It is
NOT the "flip two env vars" task the plan assumed.

Note this is separate from, and upstream of, the PIECEWISE-vs-FULL question.
We never got to measure whether PIECEWISE pays for itself, because capture
cannot start at all. The `xpu.py:223-232` FMHA downgrade (sycl-tla kernels
not capturable) remains a *second* limitation that would apply after this one
is fixed.

## Consequence for the plan

The revised priority in `DECODE_BUDGET_CORRECTED_20260812.md` ranked capture
first (~49% upper bound) and fused KDA second (~13%). **That ordering is now
inverted by feasibility, not by value:** capture is blocked behind a vLLM
distributed-layer change, while the fused KDA decode kernel is implemented,
CPU-verified, and needs only a measurement.

Capture's upper bound is unchanged and still the larger prize. It is simply
no longer the *next* thing.

## What the baseline number is worth on its own

1.154 tok/s on torch 2.11 vs 1.138 on torch 2.10 (measured 2026-08-11 on
different nodes) — the venv is worth about **+1.4%**. Small, but it would
have been silently credited to capture had I run the capture leg alone
against the old reference. The control leg earned its model load.

## Prior attempts, for the record

Attempts 1-3 on hold 8748640 and 8749119 failed for reasons that were NOT
this: dirty tiles (`--gpu-memory-utilization 0.92` unattainable), two
concurrent drivers sharing `RAY_TEMP_ROOT`, leaked Ray placement groups from
a `pkill` teardown, and a bare `wait` deadlocking on the `tee` from process
substitution. All are now fixed in the harness
(`GPU_MEM_UTIL`, `flock`, per-leg temp root, `ray stop` pre-flight,
collected-PID wait). See
`memory/feedback_kill_vs_stop_and_wait_traps.md`.
