# Async GRPO — Phase 0 + Phase 1 Implementation

This document covers the async generation + weight-sync architecture added to
the XPU GRPO recipe for Aurora. It records what shipped in Phase 0 (always-on
rollout logprobs) and Phase 1 (single-thread rollout producer with k=1
staleness), the validation runs that exercise them, and the boundary where
Phase 2 picks up.

The motivating plan is `~/.claude/plans/eventual-juggling-prism.md`. The
synchronous baseline being replaced is described in
`docs/features/vllm_weight_sync.md`.

---

## Why

The pre-Phase-1 GRPO loop was synchronous end-to-end:

```
generate (vLLM HTTP)  →  ref/policy fwd  →  fwd+bwd+opt  →  weight sync  →  next step
```

A 1-step deferred broadcast (`_start_deferred_broadcast`) overlapped the *push*
of weights to vLLM with the next step's compute, but `generate()` itself still
blocked rank 0 for the entire rollout (~3s for Qwen2.5-3B at G=8, max_gen=512).

Phase 1 lifts that block: a daemon thread on rank 0 pre-fetches the next
batch's rollout while the training ranks finish fwd/bwd/opt for the current
batch. Step time becomes `max(gen, grpo)` instead of `gen + grpo`.

Going async has one correctness consequence: rollouts produced under weights
*v* are consumed by training under weights *v+1*. The GRPO loss already
supports importance-sampling correction via
`ratios = exp(pi_logprobs - pi_old_logprobs)`, but the previous single-epoch
fast path passed `pi_old_logprobs = None` and let the loss fall back to
`pi_logprobs.detach()` — silently making the IS ratio identically 1.0.
Phase 0 closes that hole.

---

## Phase 0 — always-on rollout logprobs

### Code

`recipes/dev/grpo_full_finetune_distributed_xpu.py`
- New config key `always_compute_rollout_logprobs: bool`
  (auto-forced when `async_generation.enabled=true`)
- Single-epoch shortcut (`logprobs=None`) replaced with a rollout-time policy
  forward whose result is stored in `GRPOTrajectory.logprobs`
- Sanity assert: if a stale rollout reaches `grpo_step` with no logprobs, fail
  loudly rather than silently returning `ratios ≡ 1`

### Verification

3B sync baseline rerun with `always_compute_rollout_logprobs=true`: loss curve,
grad_norm, ratios all matched the prior sync baseline within run-to-run noise.
The extra rollout-time forward costs ~1s on 3B and is gated for larger models.

---

## Phase 1 — single-thread RolloutProducer (k=1, server mode)

### Architecture

```
rank 0                                     ranks 1..N
─────────────────────────────────────      ─────────────────────────────────
RolloutProducer thread:                    (no producer; consume only)
  loop:
    batch = next_batch()
    payload = vllm_http_generate(batch)    ─── consume time ──────────────
    queue.put((batch, payload, w_ver))      broadcast(batch, payload)
                                            ref/policy fwd  (collective)
                                            fwd + bwd + opt
                                            weight sync (deferred bcast)
                                            wsync.complete → bump w_ver
```

The producer overlaps the vLLM HTTP roundtrip with the previous step's
fwd/bwd/opt. With `vllm_weight_sync_interval=k`, the producer's effective
staleness is `k` even when the queue is `maxsize=1`.

### Why server mode only

Dedicated-rank vLLM (`_generate_with_dedicated_vllm`) calls
`broadcast_object_list` over `_training_pg`, which requires every training rank
to participate. That is incompatible with rank 0 producing solo. Phase 1
targets `vllm_mode == "server"` (HTTP) where rank 0 calls `requests` directly.
The dedicated-rank refactor (gloo ring buffer, vLLM-side dataloader replica)
is Phase 2.

### Files

| File | Role |
|---|---|
| `torchtune/dev/rl/async_rollout.py` | `RolloutItem`, `WeightVersionTracker`, `RolloutProducer` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Config plumbing (~1237), producer wiring (~772, ~2128), consume-time broadcast in `train()` |
| `torchtune/dev/grpo/loss.py` | `epsilon_high` knob (Phase 2 deliverable, see below) |
| `recipes/configs/dev/production/qwen3B_grpo_async_xpu.yaml` | Prototype config |
| `experiments/async_grpo/run_phase1_async.sh` | Launcher wrapper |

### Config surface

```yaml
# qwen3B_grpo_async_xpu.yaml
always_compute_rollout_logprobs: true   # Phase 0
vllm_weight_sync_interval: 1            # k for the deferred broadcast
async_generation:
  enabled: true                          # Phase 1 producer thread
  max_staleness: 1                       # queue capacity
loss:
  _component_: torchtune.dev.grpo.loss.GRPOLoss
  epsilon: 0.2
  # epsilon_high: 0.28                   # Phase 2 (asymmetric DAPO clip)
  kl_coeff: 0.1
```

`async_generation.enabled=true` implies `always_compute_rollout_logprobs=true`
and refuses the silent `.detach()` fallback in the loss path.

### Validation runs (Qwen2.5-3B, single Aurora node, gene-recall task)

All runs on `qwen3B_grpo_async_xpu.yaml`, G=8 fbs=4 max_gen=512, 10 training
tiles + 1 vLLM tile + 1 idle, SHM weight sync.

| Run | Steps | Steady step time | Ratios range | Clipfrac | Notes |
|---|---|---|---|---|---|
| Phase 1 smoke | 5 | 22.6 s | 1.0001–1.0007 | ≤ 0.0010 | producer ≈ 9s, fully hidden behind 18.6s grpo |
| Phase 1 stability | 20 | 22.3 s | 0.9998–1.0010 | ≤ 0.0011 | one mild kl_loss blip step 13, self-corrected |
| Phase 1.5 (k=2) | 10 | 21.4 s | 1.0000–1.0011 | ≤ 0.0019 | wsync skipped on alternating steps; k=2 doesn't bite at 3B/low LR |
| Phase 2 (k=2, ε_high=0.28) | 10 | 20.5 s | 1.0001–1.0011 | ≤ 0.0007 | identical magnitudes to symmetric k=2 |
| Phase 2 scout (k=4, ε_high=0.28) | 10 | 21.0 s | 0.9997–1.0010 | ≤ 0.0007 | even k=4 keeps π_old ≈ π_new |
| Phase 1 convergence | 50 | 20.7 s | 0.9998–1.0014 | — | clean exit; 2 ckpt saves |

### Step breakdown (5-step smoke, indicative)

```
gen     = 2.8s  (consumer pop — full rollout was already overlapped)
grpo    = 18.6s (fwd + bwd)
wsync   = 1.1s  (SHM gather; copy+http overlapped)
other   = 0.1s
total   = 22.6s
```

Producer latency (8.4–9.5 s) is fully hidden behind 18.6 s of grpo.

### Phase 2 deliverable that landed early — `epsilon_high`

`torchtune/dev/grpo/loss.py` (`GRPOLoss` and `GRPOCompletionLoss`) now accept
`epsilon_high: Optional[float]`, threaded through OmegaConf. When set, the
clamp becomes `[1 - epsilon, 1 + epsilon_high]` (DAPO-style asymmetric).
Default behavior (symmetric `[1-ε, 1+ε]`) is preserved when unset.

This is *correctness insurance* for higher-LR or smaller-model regimes. On
Qwen2.5-3B at the current LR, ratios stay so close to 1.0 that ε_high is a
no-op — the validation above proves it costs nothing when you're already
on-policy.

---

## What Phase 1 does *not* do

These items remain explicit Phase 2 work:

- **Continuous producer loop** — vLLM rank running its own dataloader replica,
  pushing into a multi-slot queue. Not needed for 3B convergence (Phase 1's
  pipelined version with `wsync_interval=k` already gets the throughput win).
  Becomes interesting when (a) gen ≫ grpo (larger models), (b) higher LR /
  faster-moving policies, or (c) the dedicated-rank vLLM path.
- **Telemetry** — `weight_version_lag`, `producer_idle_ms`, `producer_wait_ms`
  are tracked internally on the producer but not yet emitted in the metrics
  line. Defer until a workload actually saturates the queue.
- **Failure handling** — producer exceptions propagate via `RolloutProducer.get`,
  but there's no queue-staleness watchdog yet.

---

## Phase 2 Step 1 — dedicated-rank gloo ring buffer (validated 2026-04-30)

### What landed

The dedicated-rank vLLM path now broadcasts weights over the existing
`_wsync_pg` (2-rank gloo, `[0, vllm_rank]`) without dragging the rest of
the training ranks through a `broadcast_object_list` on the world PG.
Generation requests still use `_training_pg` (sized to 11 = N-1) for the
fanout to consumers; only the wsync collective was lifted off the world PG.

The architectural change that mattered for FSDP2:
- `_compute_wsync_layout` now records the **global** (unsharded) shape /
  numel for DTensor params, not the local shard shape.
- `_sync_dedicated_vllm_weights` detects FSDP2 via DTensor isinstance and
  replaces `summon_full_params` with a `_NullCtx`. The chunk-pack loop
  calls `param.full_tensor()` collectively across the sharding mesh
  (`_training_pg`), then only rank 0 packs the result into the chunk
  buffer and broadcasts to vLLM.
- `_compute_wsync_layout` lazy-builds `_tune_to_hf_map` so the dedicated
  setup path doesn't need an explicit call (server / colocate paths still
  build it eagerly in their own setup).

### Files modified

| File | Role |
|---|---|
| `torchtune/dev/rl/weight_sync.py` | `_NullCtx`, FSDP2 detection, collective full_tensor pack, lazy tune→HF build, rank-gated `get_backend` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | FSDP2 dp_mesh from `_training_pg` (DeviceMesh.from_group), `_gen_pg` send/recv on dedicated-rank generation, shutdown sentinel |
| `experiments/async_grpo/run_phase2_dedicated.sh` | Launcher (11 train + 1 vLLM, gloo loopback) |
| `recipes/configs/dev/production/qwen3B_grpo_async_dedicated_xpu.yaml` | Dedicated-rank async config |

### v10 validation run (Qwen2.5-3B, 5 steps, 11 train + 1 vLLM dedicated)

| step | loss | ratios | clipfrac | grad_norm | wsync rank=0 |
|---|---|---|---|---|---|
| 1 | -0.144 | 1.0009 | 0.0002 | 0.71 | 5.09s |
| 2 | -0.047 | 1.0000 | 0.0010 | 0.72 | 4.80s |
| 3 | +0.061 | 1.0005 | 0.0004 | 0.43 | 4.87s |
| 4 | -0.078 | 1.0013 | 0.0005 | 1.34 | 4.77s |
| 5 | -0.079 | 1.0011 | 0.0005 | 0.53 | 4.99s |

5/5 steps clean, exit=0. Ratios in [1.0000, 1.0013] match Phase 1 server-mode
parity (was [0.9998, 1.0010], clipfrac ≤ 0.0011). Wsync sender side: ~5s
total (pack 2.6s + d2h 0.13s + bcast 2.0s @ 3 GB/s gloo loopback). Receiver
total includes idle wait for sender; the actual transfer time is the same
~5s.

### Bugs fixed during Step 1 bring-up

1. **DTensor flatten on FSDP2 wsync** (v7) — `param.data.view(-1)` failed
   on unevenly sharded DTensors. Fix: collective `full_tensor()` on every
   training rank, only rank 0 keeps the materialized tensor.
2. **`get_backend(_wsync_pg)` from non-members** (v8) — `_wsync_pg` is
   `[0, N-1]`, but ranks 1..N-2 entered the wsync function and called
   `get_backend` on it → "Invalid process group specified". Fix: gate the
   `get_backend` call on `self.rank == 0`; non-members never touch the PG.
3. **`tok_embeddings` not in vLLM model** (v9) — `_tune_to_hf_map` was
   never built on the dedicated-rank setup path. Fix: lazy build inside
   `_compute_wsync_layout` if the attribute is missing.

### v12 stability run (Qwen2.5-3B, 20 steps, 11 train + 1 vLLM dedicated)

20/20 steps clean. Ratios in [0.9998, 1.0014], clipfrac ≤ 0.0017,
grad_norm 0.34–37.83 (transient spike at step 13 cooled by step 16, no
divergence). First non-zero successes at step 12 (0.006) and step 19
(0.017) — policy is responding to reward signal. Wsync rank=0 ~4.7s/step
steady-state @ 3.2 GB/s gloo loopback.

| step | ratios | clipfrac | grad_norm |
|---|---|---|---|
| 1   | 1.0000 | 0.0000 | 0.56 |
| 5   | 1.0013 | 0.0005 | 0.70 |
| 10  | 1.0001 | 0.0010 | 1.59 |
| 12  | 1.0008 | 0.0000 | 24.78 |
| 13  | 1.0005 | 0.0002 | 37.84 |
| 16  | 1.0001 | 0.0007 | 3.09 |
| 20  | 0.9998 | 0.0007 | 0.79 |

### Step 4 (shutdown sentinel) — fixed (v13 validated)

The vLLM rank's count-based `for step in range(num_steps)` loop in
`weight_sync.py:_run_vllm_generation_server` exited before rank 0's
sentinel arrived, producing a benign "Failed to send vLLM shutdown
sentinel: Connection closed by peer" warning (observed in v10 and v12).

Fix: switch to `while True:` with the sentinel as the only exit condition
(weight_sync.py:551). The vLLM loop now blocks indefinitely on the next
recv until either a real payload or `{"shutdown": True}` arrives —
eliminating the race.

v13 verify (3 steps): exit=0, "shutdown sentinel received at step 3"
logged on rank 11, ZERO "Failed to send" warnings. Confirmed in v14 too
(3 steps, exit=0, sentinel received).

### Step 3 (telemetry) — instrumentation in place (v14 non-regression)

Producer-side:
- `RolloutProducer._time_blocked_on_put_s` accumulates time the producer
  thread spent blocked on a full queue (back-pressure signal).
- `RolloutProducer._time_get_wait_s` accumulates time the consumer spent
  in `.get()` waiting for the next item (under-supply signal).
- Both have read+reset accessors `read_blocked_on_put_ms()` and
  `read_get_wait_ms()` so the metrics line shows per-step deltas.

Consumer-side: the rank-0 `METRICS step=…` line in the recipe now appends
`prod_qsize=N  weight_lag=K  prod_wait_ms=…  prod_idle_ms=…` whenever a
producer is active. Empty tail when `_rollout_producer is None` (e.g.
dedicated-rank mode where the producer is bypassed).

v14 dedicated-mode 3-step run: clean exit, telemetry tail correctly empty,
no regressions vs v12/v13.

### Open

- 32B 2-node dedicated-rank async smoke. Requires (a) new launcher with
  `mpiexec` across 2 nodes, (b) new config (22 train + 2 vLLM TP=2 ranks
  on the second node), (c) generalizing `_setup_dedicated_vllm_rank` for
  the 2-node PG topology, (d) gloo backend tuned for cross-node (hsn0
  not loopback). Defer until 32B-specific work allocates a 2-node block.
- Server-mode telemetry validation (v15+). Requires HTTP vLLM
  pre-launched; instrumentation already in place and verified
  syntactically + as a no-op on dedicated mode.

---

## Critical platform fix that made all of this run

`recipes/dev/_usercustomize_vllm/usercustomize.py` patches the vLLM registry
subprocess (Patch 2) via a lazy `__import__` hook. An eager
`import vllm.model_executor.models.registry` at top level of usercustomize.py
silently kills the vLLM EngineCore spawn child during interpreter startup —
the parent reports `Failed core proc(s): {}` with no logs. The lazy hook waits
until `vllm.config` is being loaded and only then patches the in-flight
registry module. Same pattern is used for the XPU memory patch (Patch 3).

See `feedback_usercustomize_eager_vllm_import.md` (memory) for the full
investigation. Any future async work that touches usercustomize must preserve
the lazy hook structure.

---

## How to run

Single-node, debug queue:

```bash
qsub experiments/async_grpo/hold_phase1.sh         # 1-node, 1h
ssh <allocated_node>
cd /lus/flare/projects/ModCon/ngetty/torchtune
bash experiments/async_grpo/run_phase1_async.sh 50  # 50 steps
```

Sweep `vllm_weight_sync_interval=k` to scout staleness on the same config:

```bash
bash recipes/dev/run_grpo_vllm_xpu.sh 1 10 \
    /lus/flare/projects/ModCon/ngetty/models/qwen2_5_3b 10 \
    recipes/configs/dev/production/qwen3B_grpo_async_xpu.yaml \
    async_generation.enabled=true \
    async_generation.max_staleness=2 \
    vllm_weight_sync_interval=2 \
    loss.epsilon_high=0.28
```
