# Whole-graph compile A/B — result (job 8750347, 2026-08-12)

Ran the top-of-plan lever from `PLAN_C1_ROOT_CAUSES.md` §1: attack the 61%
dispatch term with `torch.compile` at `cudagraph_mode=NONE` (which does not
route through the blocked `graph_capture()` path).

## Measured

| leg | env | tok/s (best of 3) | ms/token |
|---|---|---:|---:|
| `eager_base` | — | 1.158 | 864 |
| `compile_seg` | `ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE` | **1.395** | **717** |
| `compile_whole` | + `SPLITTING_OPS_EMPTY=1` | *(in flight)* | |

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
