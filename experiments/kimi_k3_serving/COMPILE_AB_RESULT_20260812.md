# Whole-graph compile A/B — result (job 8750347, 2026-08-12)

Ran the top-of-plan lever from `PLAN_C1_ROOT_CAUSES.md` §1: attack the 61%
dispatch term with `torch.compile` at `cudagraph_mode=NONE` (which does not
route through the blocked `graph_capture()` path).

## Measured

| leg | env | tok/s (best of 3) | ms/token |
|---|---|---:|---:|
| `eager_base` | — | 1.158 | 864 |
| `compile_seg` | `ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE` | **1.395** | **717** |
| `compile_whole` | + `SPLITTING_OPS_EMPTY=1` | 1.383 | 723 |

**+20.5%**, and tight across reps (1.363 / 1.395 / 1.384) — not node noise.

Baseline is eager with the broken fused-KDA kernel OFF, so this is measured
against trustworthy numerics, not against the retracted 1.267.

## Compile actually engaged — checked, not assumed

`analysis/compile_diagnose.sh` exists because "compile made no difference"
and "compile never ran" look identical in a tok/s number. Here they separate
cleanly:

| | `eager_base` | `compile_seg` |
|---|---|---|
| resolved mode | `CompilationMode.NONE` | `CompilationMode.VLLM_COMPILE` |
| `splitting_ops` | `[]` | `['vllm::unified_attention_with_output', 'vllm::unified_mla_at…']` |
| Dynamo transforms | 0 | 3 |
| graph breaks | 0 | 0 |
| silent eager fallback | none | none |

`Compiling a graph for compile range (1, 2048)` takes **264–280 s** — a
real one-time warmup cost, paid before the timed requests.

### `leg_env_in_worker: NOT-FOUND` on this leg is a FALSE ALARM

The check greps worker logs for the leg's first env var. That var is
`ENFORCE_EAGER`, which the **launcher** consumes to decide whether to pass
`--enforce-eager`; it is deliberately not exported to the Ray workers and so
can never appear there. The check is still worth keeping for
`VLLM_KIMI_*`-style flags that *are* copied via
`VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` — but for launcher-consumed vars the
load-bearing evidence is the resolved `CompilationMode` above, not the grep.
Do not "fix" the flag by exporting it to workers; fix the check's scope.

## Against the pre-registration

The plan pre-registered a fusion model. Decision rule was: ≥+25% main line,
+5–25% real-bank-it-then-explain, ~0 means Inductor is not fusing.

**+20.5% lands in the middle band.** Mapping onto the budget
(compute 130 ms + collectives 207 ms are untouched by fusion):

| | dispatch term | step | tok/s |
|---|---:|---:|---:|
| eager (measured) | 527 ms | 864 | 1.158 |
| **compile_seg (measured)** | **380 ms** | **717** | **1.395** |
| 3x fusion (predicted) | 306 ms | 643 | 1.56 |
| 10x fusion (predicted) | 226 ms | 563 | 1.78 |

So Inductor **is** fusing — the dispatch term fell 1.39x — but well short of
the 3x the model assumed. The residual 380 ms is now the largest single term
in the step and the obvious next target.

Note this is a *consistency check*, not a measurement of fusion: it assumes
compile changed only the dispatch term. Kineto on this stack emits no CPU
rows (`project_k3_kineto_no_cpu_rows_20260811`), so the launch count under
compile has **not** been measured directly. Do not quote "1.39x fewer
launches" — quote 1.39x on the inferred dispatch term.

## `compile_whole` is a NULL result — whole-graph compile adds nothing

**1.383 vs `compile_seg` 1.395 (−0.9%, inside rep spread 1.370/1.381/1.383).**
Engagement verified the same way: `mode=VLLM_COMPILE`, `cudagraph_mode=NONE`,
`'splitting_ops': []` — this really was one whole-model graph, not 93
attention-split segments.

This **answers the "why is compile only +20.5%" question in the section
above, and rules out its first hypothesis.** The 93 forced graph cuts at
attention boundaries were the leading suspect for limited fusion; removing
them entirely changes nothing. So Inductor was already extracting essentially
all the cross-op fusion available *within* a layer, and there is no
significant fusable work spanning layer boundaries.

Consequence for the plan: **whole-graph compile is closed as a lever.** The
remaining compile-side ideas are `custom_ops` (let Inductor own ops currently
opaque to it) and `max-autotune`.

### Side finding: plan item §4 (sequence parallelism) is blocked in platform code

`xpu.py:238-257` force-disables eight fusion passes with "not yet supported
on XPU": `enable_sp`, `fuse_gemm_comms`, `fuse_allreduce_rms`,
`fuse_norm_quant`, `fuse_act_quant`, `fuse_attn_quant`, `fuse_act_padding`,
`fuse_rope_kvcache`.

Two of those — **`enable_sp` (sequence parallelism) and `fuse_allreduce_rms`
(AllReduce+RMSNorm fusion)** — are precisely the mechanism plan §4 proposes
for the 24% collectives term. §4 is therefore not "untouched", it is
**blocked by an unconditional platform override**, and any attempt to run it
must first decide whether that override is a real capability gap or the same
blanket conservatism already found to be wrong on the graph-with-comms gate
(the `VLLM_XPU_ALLOW_GRAPH_WITH_COMMS` note in the same file says that gate
"has been shown not to hold universally").

Measured, so this is not inferred from source alone: the eager leg logged
exactly two of these warnings (`Activation + quant fusion`, `RMSNorm + quant
fusion`) and the compile legs logged **zero** with `pass_config: {}`. So the
SP / allreduce-RMS passes were never requested in any leg — they are off by
default *and* would be overridden if turned on. Nothing measured here says
they would help; it says the experiment cannot be run without touching
`xpu.py`.

## NOT YET A BANKABLE RESULT — correctness is pending

Both legs ran the default `'a'`-repeat prompt and both answered with a run
of `'a'`s:

```
COMPLETION=IDENTICAL_BUT_DEGENERATE 6 pairs; output is a single repeated
character, so this does NOT validate numerics.
```

On this exact harness the fused-KDA kernel measured +9.8% and was
**numerically wrong** (198ac00a) — caught only when a real prompt replaced
the `'a'`s. A speedup on an unvalidated path is not a result.

`rerun_compile_correctness.sh` re-runs eager vs compile with the 12-token
`17 times 23` prompt (the only prompt known to produce non-degenerate K3
text) and diffs the completions. Compile changes fusion, not semantics, so
at `temperature=0` the two legs must be byte-identical. That gate must pass
before the +20.5% is quoted anywhere as a win.

## Incidental: the corr3 data independently re-confirms the fused-KDA defect

Re-reading `logs/abc1_8749725/corr3_*` (already the basis of 198ac00a):
base answers `391 / 414 / 437` (all arithmetically correct), fused answers
`391 / 391 / 391`. Same prompt, same seed, temperature=0, one flag apart.
No new claim — noted because it is the cleanest worked example of why the
degenerate-prompt gate exists.
