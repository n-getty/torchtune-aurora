# RESULTS DISCIPLINE — the human gate for runtime numbers

**One rule:** No runtime number (step time, tokens/s, "X is faster than Y") enters
`docs/status.md`, `memory/`, a report, or a conclusion until it has passed the three
checks below. This is the `verification-before-completion` discipline, but for
**measurements**, not code. CPU tests passing means the code is correct; it says
**nothing** about whether the run executed in the mode you think it did.

## Why this exists — the 2026-06-17 incident

A Qwen3-4B full-FT GRPO run reported **274s/step** and the conclusion *"LoRA wins on
step time"* was published. It was wrong. The dense run had silently taken the
`CHUNKED_BACKWARD` path, which lacks the `_orig_reduce_scatter_tensor` bypass that
`SINGLE_BACKWARD` has — so every `reduce_scatter` went through the gloo CPU-bounce
(D2H → gloo AllReduce → H2D), adding ~130s/backward. The LoRA leg was unaffected
(adapter grads are FSDP `ignored_states`, never traverse the patched path). The two
legs ran in **different execution modes**: apples-to-oranges. CPU tests passed the
whole time. A 4B at 274s is ~4× a 32B dense — physically impossible, and the user
caught it by **monotonicity** before any tool did.

See `memory/project_lora_vs_fullft_4b_parity_20260617.md`.

## The three checks (ALL required before a number is trusted)

### (a) Monotonicity sanity-bound vs a known baseline
A smaller-or-equal model at the same topology **cannot** be slower than a larger one.
Bound every number against `docs/status.md` anchors before believing it:

| Model            | Known-good step time | Source                        |
|------------------|----------------------|-------------------------------|
| AGPT-2B (2N)     | ~13s/step            | status.md 2026-06-13          |
| Qwen2.5-3B       | ~21s/step            | status.md                     |
| dense 4B         | low-tens (LoRA ~54.5s)| extrapolated; must be < 32B  |
| Qwen3-8B         | ~27s (colocate)      | status.md                     |
| Qwen3-30B-A3B    | ~54.8s/step (G=8)    | status.md                     |
| Qwen3-32B (2N)   | 33–67s/step          | status.md                     |

A 5× anomaly is a **bug signal, not a result**. Run:
```bash
scripts/check_run_health.sh --baseline 4b 274     # WARN: 274 > 32B ceiling -> investigate
```

### (b) `check_run_health.sh <logfile>` → GREEN
The gate greps the run log for silent degraded modes and exits non-zero on any:
- gloo CPU-bounce `reduce_scatter` active on a **non-EP** run (the incident: `v206`
  PG built + `CHUNKED_BACKWARD` + `ep_degree=1` → no bypass → +130s/bwd);
- `varlen=requested-but-skipped` (a "varlen speedup" claim that silently no-op'd);
- `banned:1` / `UR_RESULT_ERROR_OUT_OF_RESOURCES` / `SIGABRT` (crash);
- no `TIMING step=` lines (the run never completed a step — the number is fabricated);
- it also prints which `grpo_step path:` was taken (SINGLE_BACKWARD / CHUNKED_BACKWARD / PACKED).

```bash
scripts/check_run_health.sh experiments/.../run.log   # must print GREEN, exit 0
```

### (c) For A/B: both legs verified same path AND same transport
The exact mistake on 2026-06-17 was comparing a leg that bypassed gloo against one
that didn't. Never trust an A/B delta until:
```bash
scripts/check_run_health.sh --compare logA logB       # FAILS loudly on path/transport mismatch
```
If the legs differ in `grpo_step path` or reduce_scatter transport, the comparison is
invalid — re-run both in the same mode.

## Checklist (paste into the PR / status entry that reports a number)

- [ ] Number is monotonic vs the closest `status.md` baseline (smaller ≤ larger).
- [ ] `check_run_health.sh <log>` → **GREEN** (exit 0), output pasted.
- [ ] `grpo_step path:` recorded (which backward path produced the number).
- [ ] If A/B: `check_run_health.sh --compare logA logB` → parity OK.
- [ ] If a feature was claimed to help (varlen, compile, EP-XCCL): the engage marker
      is in the log (`varlen=engaged`, not `requested-but-skipped`).

A number that fails any box is **provisional** — do not let it drive a decision.
