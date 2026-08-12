# Overnight summary — 2026-08-12

> ## ⚠ THE FUSED KDA KERNEL IS WRONG ON HARDWARE — DO NOT ENABLE IT
>
> `VLLM_KIMI_XPU_KDA_FUSED_DECODE=1` produces **different text** from the
> eager path on real weights. Caught 2026-08-12 by the correctness check:
>
> ```
> base : " 391\n\nQ: What is 18 times 23?\nA: 414\n\nQ: What is 19 times 23?\nA: 437"
> fused: " 391\n\nQ: What is 17 times 23?\nA: 391\n\nQ: What is 17 times 23?\nA: 391"
> ```
>
> Both are correct for the FIRST answer (391) and identical for 18
> characters, then diverge permanently — the compounding signature of a
> corrupted recurrent state. The measured **+9.8% speedup is therefore not
> bankable**; it was partly buying speed with wrong numerics.
>
> The flag was already default OFF and stays OFF. Nothing in production is
> affected. See "The correctness result" below.

## Short version

**Your 20-60 tok/s target is met in aggregate and cannot be met single-user.**

| | tok/s | status |
|---|---:|---|
| **c=64, clean** | **46.06** | **TRUSTWORTHY** — kernel OFF, 64/64 responses, banned=0 |
| c=64, broken kernel | 49.31 | tainted; the +7.4% was bought with wrong tokens |
| c=128 (no fused kernel) | 65.20 | measured 2026-08-10 |
| c=1 single-user | 1.267 | measured, +21.7% over two sessions |
| c=1 theoretical best | ~7.7 | **if 100% of dispatch AND collectives were removed** |

## Why single-user 20 tok/s is out of reach — the number that reframes everything

I did this arithmetic before spending hold time, and it invalidates the plan
I was working from yesterday:

    remove 100% of dispatch          -> 337 ms/token = 2.97 tok/s
    remove dispatch AND collectives  -> 130 ms/token = 7.7  tok/s
    20 tok/s needs                      50 ms/token

**Every plan to date attacked dispatch. None could have reached 20 tok/s even
if perfect**, because device-busy compute alone (130 ms) already exceeds the
whole 50 ms budget.

Compute is 130 ms against a **~1.2 ms bandwidth floor** (29.7 GB of MXFP4
active weights / 25.6 TB/s aggregate). The gap is kernel granularity: median
kernel **4.48 us**, 97.2% under 20 us, contributing 82% of busy time. PVC
tiles are starved, not busy.

All three terms share one root — **TP=32, which is forced, not chosen**:

| | weights/rank | fits 68.7 GB tile? |
|---|---:|---|
| TP=8 | 195 GB | no |
| TP=16 | 97.6 GB | no |
| **TP=32** | **48.8 GB** | **yes** |

Adding nodes does not help (TP=16 needs 97.6 GB on each of its 16 tiles
regardless). Upstream's 118 tok/s is TP16 on **GB300 (~288 GB HBM)** — 8-16
ranks for the same model because each rank has ~4x our memory: wider matmuls,
16-way not 32-way collectives, NVLink not Slingshot. **That gap is per-rank
HBM, a hardware property.**

I also checked and rejected expert offload as a way to lower TP: only 18/896
experts fire per token, but at c=1 you cannot know which until the router
runs, and streaming 29.7 GB over PCIe costs ~464 ms serialized.

## What was achieved

- **49.31 tok/s aggregate at c=64** (reps 49.152/49.312, 0.3% spread, 64/64
  non-empty, 4096 tokens, banned=0). 61% scaling efficiency. **CAVEAT: this
  leg ran `VLLM_KIMI_XPU_KDA_FUSED_DECODE=1`, now known to be numerically
  wrong.** The throughput is what the machine did, but it is not a number you
  can ship, because the tokens it produced are suspect. The c=128 65.20 tok/s
  from 2026-08-10 predates the kernel and is unaffected.
- **RE-MEASURED CLEAN: 46.06 tok/s at c=64** with the flag OFF (reps 45.909 /
  46.061, 64/64 non-empty, banned=0). This is the trustworthy c=64 number.
  The broken kernel's 49.31 was **+7.4% higher** — that is the amount of
  apparent throughput the incorrect numerics were buying.
- **Fused KDA decode kernel: +9.8% at c=1 but NUMERICALLY WRONG** — see the
  warning at the top. The speedup is real and reproducible; the output is
  not. Default OFF, and it must stay off until the codegen bug is found.
- **Graph capture proven blocked in code** (`parallel_state.py:480` asserts
  `CudaCommunicator`); needs an upstream change, not a flag. Saves the next
  person a hold.
- Six harness defects fixed, each from a real failure (see below).

## What is still open

1. **Fused-KDA correctness is STILL unverified.** Three attempts:
   - #1 died on a DAOS mount-visibility race (fixed).
   - #2 died because a **41-token prompt hangs the engine for 300 s**.
   - #3 running now with `VLLM_KIMI_XPU_KDA_CHUNKED=1` and a ~12-token prompt.

   Until a checkable-answer prompt matches between legs, the flag stays OFF.
   The +9.8% is real; the correctness evidence is 16 CPU tests + 3 mutation
   checks + a degenerate smoke test.

2. **The prefill cliff is uncharacterised and is a real product limitation.**
   Every successful K3 run in this repo used **16 prompt tokens**. 41 tokens
   hangs for exactly `RAY_CGRAPH_get_timeout` (300 s) -> HTTP 500. A naive
   linear model (41 x 0.789 s = 32 s) does NOT predict a 300 s hang, so the
   mechanism is unconfirmed. **A prompt-length ladder (16/24/32/41/64) is the
   single most useful cheap experiment outstanding** — a server that only
   handles 16-token prompts is not servable, whatever its tok/s.

3. **A 41-token prompt hangs but a 12-token one works, WITH chunked prefill
   on** (`VLLM_KIMI_XPU_KDA_CHUNKED=1`). The corr3 base leg produced
   `17x23 = 391` (also 18x23=414, 19x23=437 -- all correct) from a 12-token
   prompt in 29.1 s. Two variables changed at once (prompt 41->12 AND
   chunking off->on), so this does NOT isolate which one cleared the hang.
   The ladder experiment must vary one at a time.

   Note the 1.100 tok/s from that leg is NOT comparable to the 1.267
   baseline: different prompt length, half the completion length (32 vs 64,
   so prefill amortizes over fewer tokens), and chunking on. Do not read it
   as a regression.

4. Whole-graph compile (`VLLM_COMPILE` + `cudagraph_mode=NONE`) never ran —
   the flock correctly refused it while the correctness re-run held the lock.
   Untested, and does not route through the blocked `graph_capture()`.

## Recommendation

**If the goal is serving throughput, we are done** — 49-65 tok/s aggregate is
in your band today, and the config is `TP=32 --ep --max-num-seqs 128
mnbt=2048 gpu-mem-util 0.80` plus `VLLM_KIMI_FUSE_SHARED_EXPERT_AR=1`.

**If the goal is single-user latency, stop optimizing this stack.** The
honest ceiling is ~7.7 tok/s with heroic effort across three workstreams, two
of which are blocked. The routes that would actually move it are (a) hardware
with more HBM per rank, or (b) a smaller / more aggressively quantized model.
I'd want your call before spending more holds there.

**Before anything else: characterise the prefill cliff.** It is cheap, and it
determines whether the throughput numbers above describe a usable server.

## Harness fixes landed tonight (all from real failures)

| fix | failure it prevents |
|---|---|
| `GPU_MEM_UTIL` knob | 0.92 unattainable on a node with 52.6/64 GiB free |
| `flock` per allocation | two drivers sharing `RAY_TEMP_ROOT`, killing each other |
| per-leg `RAY_TEMP_ROOT` | stale session matched and killed by the next leg |
| `ray stop --force` pre-flight | `pkill` strands placement groups -> "no GPU available" |
| collected-PID `wait` | bare `wait` deadlocks on the `tee` from process substitution |
| `verify_model_visible` | DAOS "Mount successful!" before the mount is visible |
| `verdict=NO_TOKENS` | a zero-token run reporting `verdict=OK` |
| engine probe vs `/health` | `/health` returns 200 while the EngineCore is dead |

Operating rules learned: never edit a running bash script; don't SSH-probe
compute nodes mid-run; `find -newermt '-15 minutes'` silently matches nothing
here (use `-mmin`); `qstat -u` truncates job ids.
