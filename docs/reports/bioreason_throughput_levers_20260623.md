# BioReason 4B GRPO throughput levers — HW verdict (2026-06-23)

## Question
Close the gap between 4N faithful-envelope step time (~120-200s, gen-dominated) and the
old "45-53s baseline".

## Two levers tested, same-node A/B (variance-immune), 4N HSDP, max_gen=1024
1. **Replica engine bands** (job 8556851): give each of 3 HSDP replica leaders its own
   disjoint band of vLLM engines (the old code piled all 3 on engines 0-3, idling 4-11).
   - Result: **NO-OP.** bands-ON gen == baseline (~92-125s/step). Generation is not
     dispatch-bound — a 1024-token decode costs the same regardless of engine spread.
2. **stop_token_ids -> vLLM** (job 8556907): the recipe built self._stop_token_ids for
   train-side truncation but never sent stop tokens to vLLM.
   - Result: **MINOR ~5%.** Step-aligned (same seed) gen: OFF 125.5/91.2/116.2/133.3 vs
     ON 118.7/87.6/109.6/127.4 -> ON ~5-7s faster every pair (systematic, not noise).
     vLLM already stops at its built-in eos by default (stopOFF stop_rate~0.5); explicit
     stop tokens only catch the few seqs emitting <|im_end|>(151645) that the configured
     eos <|endoftext|>(151643) misses. Keep ON (free, never worse).

## Verdict
Neither lever closes the gap. Generation is **reasoning-length-bound** (len_mean ~600-715,
~half the rollouts run toward the 1024 cap, trunc_rate ~0.5) AND straggler-bound (per-step
gen 88-140s). The ~3x vs the old baseline is the **faithful-envelope cost** (2048 protein /
200 GO context, the inputs that made reward learnable) + long reasoning traces, NOT a bug.
The old 45-53s baseline was the non-faithful short envelope (128 protein / 50 GO / fbs 4-8).

To actually cut gen would require a capability/workload lever with quality tradeoffs:
shorter-reasoning reward shaping, smaller max_gen (truncates reasoning), or async overlap
with FIXED staleness=1 (current async plateaus at staleness=2; 2N-only; disabled on the
4N HSDP prod path). All out of scope for a no-quality-cost throughput fix.

## Decision
Proceed with prod RL at the honest ~120-170s/step (4N HSDP, stop-tokens ON). The goal is
the F_max uplift vs SFT 0.414, not matching an envelope-mismatched baseline number.

## Async overlap (job 8556892) — validated as mechanism, not shipped
Producer/consumer overlap works (qsize=1, ratios~1.0), ~12 steps clean exit=0. BUT
weight_lag plateaus at 2 (intended staleness=1) with one off-policy ratios=24 spike.
2N-only (raises ValueError under HSDP dp_replicate>1). Not production-ready; revisit
staleness enforcement (producer should block on weight-version) if 2N async ever needed.
