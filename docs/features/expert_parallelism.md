# Expert Parallelism — Aurora/XPU

Two active EP efforts:

1. **Qwen3-30B-A3B EP=8/16** — production-viable; EP=8 v10f validated, EP=16 cross-node smoke validated. Active area.
2. **Gemma4 26B-A4B EP=4** — paused at v154 (router non-determinism in AC recompute). See `docs/features/moe_integration.md` for deep implementation history.

This doc focuses on: EP=8/16 Qwen3 results, the WS5–WS10 weight sync optimization arc, and the TORCHTUNE_EP_* env var reference. For EP infrastructure implementation (hook design, weight slice contract, FSDP2 layering, Gemma4 backward saga) see `docs/features/moe_integration.md`.

---

## Architecture (Qwen3-30B-A3B EP=8/16)

**Dispatch**: hook-based AllGather + ReduceScatter (not AllToAll). `ExpertParallel._token_dispatch` (pre-hook) and `_token_combine` (post-hook) in `torchtune/modules/moe/_parallelism.py`. Each MoE layer gets its own `ExpertParallel` instance — no cross-layer instance aliasing.

**FSDP2 layering** (recipe `grpo_full_finetune_distributed_xpu.py:~1815–2240`):
- Non-expert params: `fully_shard` on `dp_replicate`, `reshard_after_forward=True`
- Expert params: trivial 1-rank solo `fully_shard`, `reduce_grads=False` (silences ze_handle_manager crashes on 1-rank reduce_scatter)
- All FSDP2 `reduce_grads` suppressed; grad sync runs post-backward via `_ep_release_fsdp_unsharded_grads` helper

**EP collectives**: gloo CPU-bounce (`_GLOO_EP_PG`, separate group from `_GLOO_DP_SHARD_PG`). XCCL is NOT used in the EP dispatch path by default (XCCL+OFI deadlocked at op #259 in v148–v151). `TORCHTUNE_EP_USE_XCCL=1` opts in.

**Weight slice contract**: interleaved formula — rank R owns global experts `R, R+ep_d, R+2*ep_d, ...`. Enforced by `tests/torchtune/dev/rl/test_ep_slice_contract.py` (CPU-safe, runs on login node). Contiguous slicing silently permutes tokens against weights and must never be used.

**Topology (production)**:
- EP=8: 1 train node (8 tiles, tiles 0-7) + 1 vLLM node (TP=4, 4 tiles). dp_replicate=1, dp_shard=8.
- EP=16: 2 train nodes (8 tiles each) + colocated vLLM TP=4 on tiles 8-11 per node. dp_replicate=1, dp_shard=16.

---

## Performance Results

### EP=8 v10 series (2026-04-30)

**Config**: `recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep8_xpu.yaml`, 2 nodes, `G=4 NSTEPS=3 FBS=1`, `AdamWBf16` (CPU-side optimizer state).

| Run | Change | PRE-STEP-1 | Outcome |
|-----|--------|-----------|---------|
| v10a | G=1 NSTEPS=1, plain AdamW | n/a | PASS rc=0; loss=nan (G=1 zero-advantage artifact) |
| v10b2 | G=2 NSTEPS=1 FBS=1 (chunked path) | n/a | PASS; loss=0.0061 |
| v10c | G=4 NSTEPS=3, plain AdamW | 29.99 GiB | step-2 OOM (fp32 AdamW state = +22.8 GiB) |
| **v10e** | G=2 NSTEPS=2, AdamWBf16 | **15.77 GiB** | **PASS rc=0**; both steps finite loss |
| **v10f** | G=4 NSTEPS=3, AdamWBf16 | **15.77 GiB** | **PASS rc=0**; ~3:20/step |

**v10f per-step wall (steady-state)**:

| Phase | Time | Share |
|-------|------|-------|
| gen (vLLM HTTP) | ~5.6 s | 3% |
| grpo (fwd + bwd + grad-release) | ~108.9 s | 54% |
| &nbsp;&nbsp;– fwd | ~3.8 s | |
| &nbsp;&nbsp;– bwd | ~5.6 s | |
| &nbsp;&nbsp;– **v9 grad-release helper (gloo bounce)** | **~99 s** | |
| optimizer (AdamWBf16 on CPU) | ~85.8 s | 43% |

93% of step time is CPU-side workarounds. `TORCHTUNE_EP_GRAD_RELEASE_XCCL=1` targets this (see EP=16 Phase C below).

**Requirement**: `torchtune.dev.bioreason.optim.AdamWBf16` (CPU optimizer state); `forward_batch_size=1` for chunked accumulate path under G≥2. Plain fp32 AdamW cannot fit: PRE-STEP-1 = 29.99 GiB vs available ~48 GiB.

### EP=16 cross-node smoke (2026-05-01)

**Config**: 3 nodes (3 h), 16 train tiles, 1× TP=4 vLLM per node (tiles 8-11), `NSTEPS=3`, `vllm_weight_sync: false` (SFT-init weights; ratios=1.0000 is single-epoch sync, not a real on-policy signal).

| Phase | Opt-ins | Step 0 | Step 1 | Step 2 | grpo steady | opt | resv |
|-------|---------|--------|--------|--------|-------------|-----|------|
| A | (none) — gloo EP AG/RS + gloo grad-release | 158.2s | 148.4s | 143.3s | 81.5–87.3s | 44–55s | 41.49 GiB |
| B | `TORCHTUNE_EP_USE_XCCL=1` | 160.2s | 145.6s | 144.2s | 83.0–91.6s | 44–52s | 41.68 GiB |
| C | `TORCHTUNE_EP_USE_XCCL=1` + `TORCHTUNE_EP_GRAD_RELEASE_XCCL=1` | 109.6s | 97.3s | 99.5s | **38.0–40.2s** | 44–52s | 45.08 GiB |

Steady-state (steps 1–2 mean):
- Phase A: **145.0 s/step** (gloo baseline)
- Phase B: **144.9 s/step** — XCCL EP AG/RS alone has **no measurable win**. The gloo CPU-bounce dominance is in the grad-release helper, not the dispatch collectives.
- Phase C: **98.4 s/step** — −32% vs A. Same magnitude as EP=8 iter2.

Memory: Phase C +3.6 GiB vs A (XCCL on-device chunks), well within 64 GiB budget.

**No-sync baseline** (same config, `vllm_weight_sync: false`): ~47s/step. This confirms the EP=16 training path itself is fast — the entire 97-100s/step delta between no-sync and Phase A is the weight sync cost. Every collective-shape optimization (WS6, WS7, WS8) failed because rearranging the same bytes on the same fabric cannot change this; only WS10 (eliminating the `_shard_pg` AllGather) addresses the root cause.

**Blocker**: `vllm_weight_sync: false` in this run — true on-policy training (ratios < 1.0) requires WS10 sender-side to land first. See "WS Weight Sync Optimization" below.

---

## WS Weight Sync Optimization (EP Expert Weight Sync to vLLM)

The EP MoE weight sync bottleneck is the `_shard_pg` AllGather that reconstructs the full expert tensor before broadcasting to vLLM. At EP=16, this dominates at ~70-75s/step.

### Optimization arc (WS5–WS10)

| Attempt | What | Result | Why |
|---------|------|--------|-----|
| WS5 (baseline) | On-device AdamW | Optimizer 87.5s→0s (220×); wsync_gather=75s now dominant | Not wsync work; established the floor |
| WS6 | `TORCHTUNE_EP_WSYNC_LAYER_BATCH=1` — batch 3 projections per layer into 1 AllGather | **0% win** | Gather is bandwidth-bound, not latency-bound; fewer collectives don't help when bytes are identical |
| WS7 | `TORCHTUNE_EP_WSYNC_GATHER_ROOT=1` — replace AllGather with gather(dst=root) | **−5% regression** | XCCL `gather()` likely implements as allgather+extract on the same fabric; no byte savings for root |
| WS8 | `TORCHTUNE_EP_WSYNC_FP8_WIRE=1` — cast expert shards to E4M3 before AllGather | **Negative result** | Adds quant+dequant overhead, doubles collectives (fp8 + scale), same bytes to vLLM end; not bit-exact |
| **WS10** | Skip `_shard_pg` entirely; each EP rank broadcasts its local shard directly to vLLM | **Phase A landed** (receiver); **Phase B/C pending** (sender) | Eliminates the 70s _shard_pg AG; projected wsync_gather 75s→5s |

**Conclusion from WS6–WS8**: The `_shard_pg` AllGather floor (~0.85 GiB/s effective throughput) is fabric-bandwidth-limited. Fewer collectives or smaller collective footprint on the same fabric can't win. The only lever is eliminating the collective entirely (WS10).

### WS10 architecture

Current (WS5):
```
trainer EP rank 0     ─── all_gather ──► full expert tensor (16x bytes)
(active sender rank)       _shard_pg      │
                                          └──► XCCL broadcast ──► vLLM workers
                                                _xccl_wsync_pg    (each takes slice via expert_map)
```

WS10:
```
trainer EP rank R ─── broadcast (src=R) ──► vLLM workers (filter via expert_map, discard others)
                        _xccl_wsync_sharded_pgs[R]
```

Per-rank broadcast: 16 EP ranks each broadcast 1/16 of the bytes sequentially over the same fabric. Total bytes identical; `_shard_pg` AG vanishes. Projected total wsync: 92s→22s (gain is entirely from dropping the 70s AG).

**Expert-ownership math**: Trainer rank R owns interleaved global experts `R, R+ep_d, R+2*ep_d, ...`. Each vLLM worker checks `expert_map[global_idx]`; if owned, copies the shard into its local FusedMoE at the mapped position.

**Implementation status**:
- **Phase A** (LANDED 2026-05-03): `_load_fused_moe_experts_sharded()` in `torchtune/dev/vllm_weight_sync_worker.py`. Receiver detects tagged entries (`trainer_ep_rank` + `ep_degree` in tensors_meta) and routes to new method. Off-path is byte-identical. CPU pin-down: `tests/torchtune/dev/rl/test_sharded_vllm_moe_sync_equivalence.py` (7/7).
- **Phase B/C** (NOT YET IMPLEMENTED): sender-side wire in `_sync_weights_to_vllm_xccl`. Gated by `TORCHTUNE_EP_WSYNC_SHARDED=1` (currently no-ops with warning). Requires extending `_setup_xccl_wsync_pg` to build `_xccl_wsync_sharded_pgs[R]` (one 2-rank PG per trainer EP rank per vLLM replica).

**Implementation touch list for Phase B/C** (4 layers, ordered by dependency):
1. `_setup_xccl_wsync_pg` — add `ep_degree` per-replica PGs: `_xccl_wsync_sharded_pgs[r]` for `r in [0..ep_d-1]`
2. vLLM `init_xccl_communicator` — extend to accept `sharded_ranks` list and build receive-side PGs
3. Sender staging (`_sync_weights_to_vllm_xccl` EP branch) — skip `_shard_pg`, each rank builds its own `local_w13`/`local_w2`, populates local `tensors_meta` with tags, `all_gather_object` for manifest unification at rank 0
4. Deferred broadcast loop — iterate `ep_degree` times per batch, one per trainer EP rank, using `_xccl_wsync_sharded_pgs[R]`

**Key risk**: `_xccl_wsync_pgs` topology is 2-rank PGs (trainer src + vLLM dst pair). Each trainer rank broadcasts to its peer; the vLLM root must forward internally to other vLLM workers in the TP/EP group. Verify topology in Phase A first.

**Acceptance**: 3 clean steps EP=8 first, then EP=16. `wsync_gather` should collapse to ~5s (bf16 cast + CPU staging of non-expert params remains). `wsync_bcast_wait` stays ~22s but now covers ALL bytes including experts.

---

## Gemma4 EP=4 Backward Blockers (paused)

For full implementation history see `docs/features/moe_integration.md` §"Backward Dispatch Saga" and §"v154 Result". Summary:

**v153 (collective ordering desync)**:
After 153 versions, with gloo CPU-bounce for all EP collectives and `use_reentrant=True` AC, ranks within each EP group fell out of step on backward collective ordering. Root cause: reentrant AC schedules backward hooks in a topological order that interacts with per-rank routing imbalance — the rank with the smallest routed batch in each EP group (`local_index=1`) consistently lags one labeled-op behind its peers. Gloo matches collectives by call order; once desynced, eventual size-mismatched collective → crash.

**v154 (router non-determinism in AC recompute)**:
Switching to `use_reentrant=False` eliminated the collective desync (op #259 never triggered). New failure: `ScatterAddBackward0` shape mismatch (`±1` off on scatter dimension). `ExpertParallel._token_dispatch` saves `_ag_gather_idx`/`_ag_s_local` as instance attributes; non-reentrant AC re-runs forward during backward, overwriting those attributes. If the router (sigmoid + argsort) produces even bitwise-different `scores`, top-k assignments at tie boundaries flip → `bincount` differs by ±1 → saved `routed_output` shape mismatches recomputed `idx_exp`.

**Three fix options** (none implemented; pick before next compute spend):

1. **Cache instance attributes per AC region** (~30 LOC in `_parallelism.py`): save `_ag_gather_idx`/`_ag_s_local` before AC recompute; restore after. Doesn't fix router determinism but ensures both sides agree on the original answer. Cheapest.
2. **Save router outputs as tensors**: force AC to keep `(scores, selected_experts, num_tokens_per_expert)` from original FWD. Fixes at source but adds O(T·num_experts·layers) memory.
3. **Move EP dispatch outside autograd**: restructure dispatch/combine as side-effecting ops with manual gradient handling. Cleanest, reusable for any MoE+AC; largest diff.

**Note for Qwen3**: Qwen3MoE avoids both blockers — `Qwen3MoeTransformerLayer` puts MoE **outside** AC, so there is no router recompute. The v153/v154 blockers are Gemma4-specific. Qwen3 EP blocked differently (v1-v8 train-fwd IPC crash, resolved by v8i topology change).

---

## TORCHTUNE_EP_* Environment Variables

| Variable | Default | Effect | Status |
|----------|---------|--------|--------|
| `TORCHTUNE_EP_DEBUG=1` | off | EP forensic prints in `_parallelism.py` and `experts.py` (NTPE-AG / EP-DISPATCH / EP-COMBINE / PRE-RS-BWD / `_ep_mem_probe`). SLOW-threshold prints remain unconditional. | Use for diagnosis only |
| `TORCHTUNE_EP_USE_XCCL=1` | off | Routes `_ep_all_gather` and `_ep_reduce_scatter` through native XCCL instead of gloo CPU-bounce | Iter1: 3.9% win on v10f production. Pair with GRAD_RELEASE for stacked effect. **RISK**: v141–v150 hit OFI CQ deadlock at op #259 on this path; verify clean before any production envelope run |
| `TORCHTUNE_EP_GRAD_RELEASE_XCCL=1` | off | Routes `_ep_release_fsdp_unsharded_grads` per-FSDPParam all-reduce through native XCCL on dp_shard PG instead of gloo CPU-bounce | Iter2: **dominant lever** — per-chunk all_reduce 25s→7.6s (3.3×); v10f wall −32% vs gloo, −34.5% vs gloo baseline |
| `TORCHTUNE_EP_WSYNC_LAYER_BATCH=1` | off | Batch 3 expert projections per layer into 1 AllGather on `_shard_pg` (144 → 48 collectives for EP=16) | **0% win** — bandwidth-bound. Off-path byte-identical |
| `TORCHTUNE_EP_WSYNC_GATHER_ROOT=1` | off | Replace AllGather with gather(dst=rank 0) on `_shard_pg` | **5% regression** — XCCL gather likely implements as allgather+extract. Off-path byte-identical |
| `TORCHTUNE_EP_WSYNC_FP8_WIRE=1` | off | Cast expert shards to E4M3 before AllGather; decompress back to bf16 on active rank. ~2× fewer wire bytes on `_shard_pg` | **Negative result** — adds quant/dequant, doubles collectives, same bytes to vLLM. NOT bit-exact. Mutually exclusive with LAYER_BATCH and GATHER_ROOT |
| `TORCHTUNE_EP_WSYNC_SHARDED=1` | off | **RESERVED** — skip `_shard_pg` entirely; each EP rank broadcasts local shard (WS10 Phase B/C). Receiver landed; sender not implemented. Currently emits warning and falls through | Target: wsync_gather 75s→5s |

**Recommended production config** (EP=16, current best):
```bash
export TORCHTUNE_EP_USE_XCCL=1
export TORCHTUNE_EP_GRAD_RELEASE_XCCL=1
# Result: ~98.4 s/step (EP=16 Phase C, 3/3 clean)
```

---

## Files of Record

| File | Purpose |
|------|---------|
| `torchtune/modules/moe/_parallelism.py` | `ExpertParallel`, `_token_dispatch`, `_token_combine`, `_ep_all_gather`, `_ep_reduce_scatter`, `_ep_release_fsdp_unsharded_grads` |
| `torchtune/modules/moe/experts.py` | `GroupedExperts` BMM forward (scatter-pad-bmm-gather); `@torch.compiler.disable` guards |
| `torchtune/dev/vllm_weight_sync_worker.py` | `_load_fused_moe_experts()`, `_load_fused_moe_experts_sharded()` (WS10 Phase A receiver) |
| `torchtune/dev/rl/weight_sync.py:~1280–2730` | `_setup_xccl_wsync_pg`, `_sync_weights_to_vllm_xccl` (EP MoE path), `_start_deferred_broadcast` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py:~1815–2240` | EP mesh setup, pre-FSDP2 expert weight slicing, `_ep_plan` registration |
| `tests/torchtune/dev/rl/test_ep_slice_contract.py` | CPU-safe regression: interleaved slice formula matches dispatch ownership |
| `tests/torchtune/dev/rl/test_sharded_vllm_moe_sync_equivalence.py` | WS10 Phase A CPU pin-down (7/7) |
| `experiments/ep_parallelism/` | All EP launchers and PBS scripts |
| `recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep{8,16}_xpu.yaml` | EP=8 and EP=16 configs |
| `docs/features/moe_integration.md` | Full EP implementation history (hook design, weight slice contract, FSDP2 layering, Gemma4 v141–v154 saga, Qwen3 v1–v10 series) |
| `docs/reports/qwen3_ep16_smoke_20260501.md` | EP=16 Phase A/B/C raw results (archived in `docs/reports/archive/`) |
| `docs/reports/MoE_EP_status_ws8_ws10_design.md` | WS8 analysis and WS10 full design doc (archived) |
| `memory/project_qwen3_ep_v10_unblocked.md` | EP=8 v10 result matrix |
| `memory/project_qwen3_ep16_ws10_validated.md` | WS10 Phase A landing + receiver validation |
| `memory/project_qwen3_ep_grad_release_xccl_iter2.md` | TORCHTUNE_EP_GRAD_RELEASE_XCCL=1 results |

---

## MoE Optimization Roadmap (see also `docs/features/moe_integration.md`)

Priority-ordered (EP=16 path to production):

| Priority | Item | Status |
|----------|------|--------|
| P0 | WS10 Phase B/C sender-side wire | Not started; design complete |
| P0 | End-to-end vllm_weight_sync=true run with real reward signal | Blocked on WS10 sender |
| P1 | `use_reentrant=False` + cache `_ag_gather_idx`/`_ag_s_local` for Gemma4 | Not started; pick one fix option |
| P1 | Re-benchmark EP=1 vs EP=4 at large enough batch to amortize dispatch | No data above grpo_samples=2 |
| P2 | 3-node EP=8 dp_replicate=3 (spread optimizer state; may restore on-device AdamW) | Untested |
| P2 | Move expert AdamW back to GPU | Blocked on memory headroom (~7 GiB free at PRE-STEP) |
| P3 | WS11: fp8 on _xccl_wsync_pg broadcast path (only after WS10 numerics validate) | Design only |
