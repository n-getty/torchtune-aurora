# Exactly what to run when hold 8749119 starts

Two independent measurements are queued up, both fully prepared. Run them in
this order — capture first, since it has the larger projected upside and its
path is already proven clean end-to-end.

Get the full job id first (the harness rejects a short one):

```bash
JOB=$(qstat -u ngetty | awk '/kimi_k3_h/ && $10=="R" {print $1}')
# expand to the full form, e.g. 8749119.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
JOB=$(qstat -x -f "$JOB" | awk -F'= ' '/^Job Id/{print $2}')
```

`ab_c1_levers_3node.sh` now refuses any leg that cannot finish in the
remaining walltime (`MIN_MINUTES_PER_LEG`, default 30) — that guard exists
because attempt 3 died at 100% model load with 18 minutes left.

---

## 1. Graph capture at TP=32 (~30 min, the big one)

```bash
cd /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
PYTHON=/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python \
RAY_ENV_MODE=torch211 REPEATS=2 \
LEGS="capture=ENFORCE_EAGER=0,CUDAGRAPH_MODE=FULL_DECODE_ONLY,VLLM_XPU_ENABLE_XPU_GRAPH=1,VLLM_XPU_ALLOW_GRAPH_WITH_COMMS=1" \
bash ab_c1_levers_3node.sh "$JOB"
```

`PYTHON` and `RAY_ENV_MODE` must move together — the harness now hard-fails on
a mismatch, but the reason it exists is that a previous run silently put the
head on torch 2.11 and the 24 remote workers on 2.10.

**Verify before believing any number.** Each rank logs `K3_WORKER_GATES`; the
run is only valid if it shows `torch=2.11.0+xpu supports_xpu_graph=True
enforce_eager=False`. Expect `cudagraph_mode=PIECEWISE`, not FULL —
`xpu.py:223-232` downgrades because sycl-tla FMHA cannot be captured.

**Pre-registered decision rules** (baseline is the AR-fused default, 879 ms /
1.138 tok/s; dispatch is ~542 ms of it, of which ~428 ms is capturable):

| observed | reading | action |
|---|---|---|
| >= +15% | capture is working, PIECEWISE is not the binding constraint | pursue FULL capture — needs a capturable attention backend |
| +3% to +15% | real but PIECEWISE-bound (93 segments each paying replay) | bank it, switch to the fused KDA kernel |
| ~0% or negative | replay overhead cancels the saving at this segment count (consistent with Phase 0's 0.55-0.83x microbenchmark) | report and STOP — do not retry blindly |

Upper bound if all capturable dispatch vanished: 451 ms = 2.22 tok/s (+95%).
That will not happen under PIECEWISE; it bounds the claim.

---

## 2. Fused KDA decode kernel (~15 min, can share the same hold)

New this session. Collapses conv1d(q,k,v) + the recurrence into one Triton
launch — 60 aten ops/layer x 69 layers = 4,140 launches, ~114 ms. This is the
share that graph capture structurally *cannot* reach, so it composes with 1.

```bash
LEGS="fused_kda=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1" \
bash ab_c1_levers_3node.sh "$JOB"
```

Default venv/frameworks is correct here — this leg does **not** need torch
2.11. To measure both together (needs ~60 min):

```bash
PYTHON=/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python \
RAY_ENV_MODE=torch211 REPEATS=2 \
LEGS="capture=ENFORCE_EAGER=0,CUDAGRAPH_MODE=FULL_DECODE_ONLY,VLLM_XPU_ENABLE_XPU_GRAPH=1,VLLM_XPU_ALLOW_GRAPH_WITH_COMMS=1;capture_plus_fused=ENFORCE_EAGER=0,CUDAGRAPH_MODE=FULL_DECODE_ONLY,VLLM_XPU_ENABLE_XPU_GRAPH=1,VLLM_XPU_ALLOW_GRAPH_WITH_COMMS=1,VLLM_KIMI_XPU_KDA_FUSED_DECODE=1" \
bash ab_c1_levers_3node.sh "$JOB"
```

**Pre-registered:** projected **+13%** alone (879 -> 767 ms, 1.138 -> 1.30
tok/s), charging 4,140 removed launches at the measured 27.6 us residual rate.

| observed | reading |
|---|---|
| +8% to +18% | matches the launch-count model — the model is predictive, use it for the next lever |
| +1% to +8% | real but the residual is not purely per-launch; re-derive the per-launch rate before projecting anything else |
| ~0% | the residual is NOT per-launch host work — the whole dispatch-bound thesis needs re-examination, which is a bigger finding than the kernel |
| negative | Triton launch overhead on XPU exceeds 60 eager ops; check whether the JIT is recompiling per step |

Also confirm correctness on real weights, not just the CPU tests: the leg
sends the same fixed prompt as every other leg, so **the completion text must
match the baseline leg's**. A fused kernel that is fast and wrong is the worst
outcome, and 15 CPU equivalence tests do not prove XPU codegen.

**Watch for** `VLLM_KIMI_XPU_KDA_FUSED_DECODE resolved to True` in a *worker*
log (not just the driver) — that is the engagement check, same discipline as
the AR-fusion leg.

---

## If the hold is short

Capture needs ~25 min of cold model load before it can time anything. If less
than 30 min remain, run the fused-KDA leg instead — it loads the same but the
harness guard will tell you honestly either way rather than dying at 100%
load like attempt 3.
