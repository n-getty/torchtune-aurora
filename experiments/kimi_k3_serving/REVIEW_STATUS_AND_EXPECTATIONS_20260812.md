# Kimi-K3 on Aurora — status, expectations, and where to focus

**For external review. Written 2026-08-12.** Supersedes the scattered
`NEXT_STEPS_*` / `HANDOFF_*` notes as the single orientation document.
Every number below is measured on hardware unless explicitly labelled
*projected* or *upper bound*.

---

## 1. Headline

| workload | status | number |
|---|---|---:|
| **Aggregate throughput (many users)** | **target MET** | **65.20 tok/s** @ c=128; 46.06 @ c=64 |
| **Single-user latency (c=1)** | **target NOT met, and not reachable** | **1.395 tok/s** (compile), 1.158 (eager) |

These are two different problems with two different verdicts, and conflating
them has repeatedly produced wrong priorities. The aggregate goal is done.
The single-user goal is the open one, and **the honest expectation is that
20 tok/s single-user is not achievable on this hardware** — see §4, which
gives the arithmetic rather than an opinion.

---

## 2. What is measured and settled

### 2.1 The c=1 decode budget

Step = **879 ms** at the AR-fused default (1.138 tok/s); 864 ms on the
current eager control (1.158 tok/s). Decomposed from a real trace plus a
direct collective measurement:

| term | ms | share | how it was obtained |
|---|---:|---:|---|
| device-busy compute | 130 | 15% | trace, union of kernel intervals |
| collectives | 207 | 24% | 371 × 0.557 ms measured 14 KiB 32-rank all_reduce |
| host/dispatch residual | **542** | **62%** | remainder; ~28 µs × 19,642 launches |

Nothing is unattributed. Two earlier framings were **retracted**: the
"7.2 tok/s eager ceiling" (built on a 6.5 µs single-op microbenchmark and a
launch count that folded prefill in) and the "collectives are 37–74%"
estimate (measured at 20.6%).

### 2.2 Results this session

| lever | result | reading |
|---|---:|---|
| `compile_seg` (torch.compile, `cudagraph_mode=NONE`) | **1.395 tok/s, +20.5%** | real, engagement verified; **correctness gate still pending** |
| `compile_whole` (`splitting_ops=[]`, one graph) | **1.383 tok/s** | **NULL** vs compile_seg |
| graph capture via env flags | **NO_START** | blocked in code, definitive |
| fused KDA decode kernel | +9.8% but **numerically wrong** | reverted from the default path |

**`compile_whole` ties `compile_seg`, and that is the informative result.**
The leading hypothesis for why compile only bought 20% was the 93 forced
graph cuts at attention boundaries. Removing them entirely changed nothing,
so Inductor was already extracting the available intra-layer fusion and
there is no material fusable work spanning layers. Both legs verified
engaged (`mode=VLLM_COMPILE`, resolved `splitting_ops` read from the worker
log) — this is a null, not an unengaged run.

### 2.3 Hard blockers found in code, not inferred

- **Graph capture** — `parallel_state.py:480` asserts `CudaCommunicator`.
  `VLLM_XPU_ALLOW_GRAPH_WITH_COMMS` clears an *earlier, different* gate. Two
  gates; env flags cannot reach the second. Needs upstream per-platform
  dispatch.
- **Sequence parallelism (plan item 4)** — `xpu.py:238-257` unconditionally
  force-disables `enable_sp` and `fuse_allreduce_rms` as "not yet supported
  on XPU". Those *are* item 4's mechanism, so it is **blocked, not
  untouched**. Confirmed from logs: compile legs show `pass_config: {}`.
- **TP=32 is forced by memory** — 1.56 TB of weights vs 68.7 GB tiles.
  TP=8 would need 195 GB/rank, TP=16 needs 97.6. Only TP=32 fits at 48.8.

---

## 3. Where the 113× vs upstream actually lives

Upstream reports **118 tok/s at c=1** on 16× GB300 (TP=16). We are at 1.04–1.40.

GB300-vs-PVC hardware is worth perhaps 3–5×. The rest is structural, and the
dominant structure is **per-rank HBM**: GB300 has ~288 GB/rank, so the same
model runs at TP=8–16 where we need TP=32. Everything downstream — the
collective count, the collective width, the per-rank kernel size — follows
from that one constraint.

**This is a hardware property, not a software gap.** It bounds what any
amount of engineering on our side can recover.

---

## 4. The expectation that matters: 20× is not reachable at c=1

Every lever pursued to date attacks dispatch. Taking each term to **zero**:

```
today                            864 ms   1.16 tok/s
remove 100% of dispatch          337 ms   2.97 tok/s
remove dispatch AND collectives  130 ms   7.70 tok/s   <- 6.6x, still not 20x
20x target                        43 ms  23.20 tok/s
```

**130 ms of device-busy compute already exceeds the 43 ms budget for 20×.**
No combination of the levers on the board reaches the target even at
perfection. This should be stated plainly to reviewers rather than
accumulated in 20% increments.

### The one place slack remains

That 130 ms is **not** a hardware floor. Active MXFP4 weights are ~29.7 GB
against 25.6 TB/s aggregate → a bandwidth-bound step is **~1.2 ms**. We are
running at roughly **0.9% of bandwidth efficiency**, because the kernels are
tiny: **median 4.48 µs, with 97.2% of kernels under 20 µs contributing 82% of
busy time.** The tiles are starved, not saturated.

So the only credible route to 20× is **making the compute term efficient** —
far fewer, far larger kernels — not removing more host overhead. That is a
much deeper change than any lever currently in the plan, and it is the honest
framing of the gap.

### Options already checked and rejected

- **Speculative decoding / MTP** — K3 config has
  `num_nextn_predict_layers: 0`, and the checkpoint has zero draft tensors.
  No free self-speculation.
- **Expert offload to cut TP** — only 18/896 experts fire per token, but at
  c=1 you cannot know which until the router runs; streaming ~29.7 GB over
  PCIe (~64 GB/s) costs ~464 ms serialized. Worse than the problem.
- **More nodes** — does not reduce TP; the model still must fit per-rank.

---

## 5. In flight right now

| item | job | why it matters |
|---|---|---|
| **Correctness gate: eager vs compile, real prompt** | 8750347 (hold) | **The +20.5% is not quotable until this passes.** Both original legs used a degenerate `'a'`-repeat prompt. |
| **TP-group locality bench** | 8750456 (free queue) | Decides whether the TP/PP re-topology lever is worth a hold. |

**Correctness gate, leg 1 of 2 (as of 13:54).** `ccorr_eager` returned a real,
non-degenerate, arithmetically correct completion on the 17-token prompt:

```
" 17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391.\nQ: What is "
```

This is the reference the compile leg must reproduce byte-for-byte at
`temperature=0`. Note `ccorr_eager` reports 1.060 tok/s, below the 1.158
eager control — a different leg config (`VLLM_KIMI_XPU_KDA_CHUNKED=1`, 32
tokens vs 64), so it is a **correctness** reference, not a timing one, and
must not be compared against the §2.2 table.

### 5.1 Why the correctness gate is not optional

On this exact harness the fused-KDA kernel measured **+9.8% and was
numerically wrong** — base answered `391/414/437`, fused answered
`391/391/391` at temperature 0. It survived **16 CPU-interpret unit tests and
3 mutation checks**; only a real-prompt text diff caught it. A degenerate
prompt reproduces "identical output" for a broken path exactly as well as for
a correct one.

**Standing rule: a speedup on an unvalidated path is not a result.**

### 5.2 The TP/PP question, and a correction to the plan

The plan proposed **PP=2 × TP=16** to halve all_reduce width. That is wrong:
Aurora nodes have **12 tiles**, so a 16-rank TP group still straddles a node
boundary and every all_reduce stays on Slingshot. It cuts width on a fabric
already measured **latency-bound** (37× bytes → 1.25× time) — the same shape
as the WS6/WS7/FP8-wire levers, all three of which measured null.

The configuration the plan missed is **TP=12 × PP=3 = 36 ranks = 3 nodes ×
12 tiles**, which makes every TP group **node-local**:

| | ranks | nodes | GB/rank | TP node-local? |
|---|---:|---:|---:|---|
| TP=32 × PP=1 (today) | 32 | 2.67 | 48.8 | no |
| TP=16 × PP=2 (planned) | 32 | 2.67 | 48.8 | **no** |
| **TP=12 × PP=3** | **36** | **3.00** | **43.3** | **YES** |

Same three nodes, *more* headroom per rank. Legality checked in source:
`KimiK3ForConditionalGeneration` inherits `SupportsPP` via
`KimiLinearForCausalLM`; 96 heads % 12 == 0; `ep_size = tp_size` forces
EP=12 and 896 % 12 leaves 8, but the remainder is distributed and only EPLB
requires even division.

Job 8750456 is currently **queued behind an unrelated 16-node debug-scaling
job on the same account** (`Not Running: User has reached queue debug-scaling
running job lim`), not blocked technically. ~3 min of runtime once it starts.

**It is gated on a measurement, not adopted.** The entire value rests on one
unmeasured number — is an intra-node 12-rank all_reduce actually cheaper than
a 32-rank cross-node one at 7–14 KiB? The latency-bound finding predicts
**no**. Decision rule pre-registered before the run: **≤0.4× → pursue;
≥0.8× → drop the lever.** Running on the free debug queue precisely so a K3
hold is not spent to learn it.

---

## 6. Recommended focus, in order

1. **Close the correctness gate** (in flight, no extra cost). Either the
   +20.5% becomes real or it evaporates — it is currently the only c=1 win.
2. **Read the locality bench and act on the pre-registered rule** (in flight,
   free). Most likely outcome: drop the re-topology lever and say so, which
   removes a multi-hour bring-up from the plan.
3. **Decide the strategic question explicitly, with the customer:** is c=1 on
   K3/Aurora worth continuing? §4 says the ceiling is ~7.7 tok/s even with
   perfect software, against a 20 tok/s ask. Aggregate throughput is already
   met. This is a scoping decision, not an engineering one, and it should be
   made deliberately rather than by continuing to grind percentages.
4. **If c=1 continues, the only target worth attacking is the 130 ms compute
   term** (0.9% of bandwidth-bound, median kernel 4.48 µs). Not more dispatch
   work. This means large fused kernels across the dense path — expensive,
   and every one needs a hardware correctness gate from day one.
5. **Upstream-facing:** the capture gate (`CudaCommunicator` assert) and the
   `xpu.py` blanket fusion-pass disables are both Intel/vLLM-side. If XPU
   graph capture were unblocked its upper bound is ~62% of the step — the
   largest single lever that exists, and one we cannot reach ourselves.

---

## 7. Known traps for anyone picking this up

- **Prompt length**: every successful K3 run in this repo uses **≤16 prompt
  tokens**. A 41-token prompt hangs exactly 300.9 s (`RAY_CGRAPH_get_timeout`)
  and returns HTTP 500. Mechanism not established — do not assert a cause.
  `VLLM_KIMI_XPU_KDA_CHUNKED=1` is the mitigation (HW-validated 2.15–7.12× on
  prefill, default OFF).
- **`mnbt ≥ 4096` → `banned:1`** at KV-init, not during serving. 2048 is the
  only validated setting at TP=32.
- **CPU-interpret tests do not prove XPU codegen.** See §5.1.
- **Kineto on this stack emits no CPU rows** — launch counts cannot be read
  directly from traces; they were derived by segmenting on a per-step marker
  kernel.
- **Launch-count projections run ~1.5–1.8× optimistic** here. Two independent
  data points (fused-KDA, compile).
