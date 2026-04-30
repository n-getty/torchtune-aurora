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
- **Dedicated-rank vLLM async** — needs a gloo ring buffer to replace the
  `broadcast_object_list` over `_training_pg`. This is the architectural
  unblocker for 32B / 30B-MoE async runs.
- **Telemetry** — `weight_version_lag`, `producer_idle_ms`, `producer_wait_ms`
  are tracked internally on the producer but not yet emitted in the metrics
  line. Defer until a workload actually saturates the queue.
- **Failure handling** — producer exceptions propagate via `RolloutProducer.get`,
  but there's no queue-staleness watchdog yet.

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
