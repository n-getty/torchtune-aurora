# Project Status — Aurora RL (torchtune XPU)

Last updated: 2026-04-24

This document synthesizes the current state of the vLLM weight sync implementation,
active training runs, open issues, and prioritized next steps. It is a living companion
to `docs/features/vllm_weight_sync.md`, `docs/features/moe_integration.md`, and `docs/experiments/aurora_rl_baselines.md`.

## Where we are (one-page stock-take)

- **Production-ready paths**: Qwen2.5-3B (10+2 tiles, 21s/step), Qwen3-32B (10+2 tiles
  server mode 25.6s; 2-node HSDP 19.4s; 12 tiles training-only ~144s/step),
  Gemma4-26B-A4B EP=1 (24s/step). Use these now.
- **Weight sync**: SHM transport stable (3B fully hidden, 31B 13s waited, addressable).
  XCCL broadcast prototype validated for direct GPU→GPU sync (3.1s for 5.75 GiB) — could
  replace SHM if needed; currently optional.
- **Framework**: `frameworks/2025.3.1` is production-ready with `unset PYTORCH_ALLOC_CONF`.
  `expandable_segments:True` is an Intel oneCCL bug (USM pointer rejection) — file
  upstream; works but unusable with collectives.
- **Expert Parallelism (EP=4/DP=3 for Gemma4 26B-A4B)**: not production-ready.
  Forward works end-to-end; backward fails deterministically at op #259. v153 ruled
  out the CXI-NIC-contamination hypothesis and exposed a per-rank autograd ordering
  bug: ranks within an EP group execute backward ops in different orders, causing
  gloo collective mismatch. See `docs/features/moe_integration.md` for the v141–v153 saga.
- **Training stability (Gemma4-31B gene recall)**: KL explodes at step 4 — root cause
  is task-level (no SFT warm-up, no EOS in 512 tokens), not infrastructure. Must fix
  before any meaningful Gemma4 RL run.
- **32B GRPO backward regression (2026-04-23, FIXED)**: Commit acdc7c9f introduced a
  per-chunk fwd+bwd loop that caused +44 GiB OOM at step 0 backward (FSDP2 unsharded
  grads kept live across chunks). Fix: `TORCHTUNE_USE_CHUNKED_LOSS=1` gate for
  single-backward path. The gloo reduce_scatter patch (added for EP) also added 130s
  to non-EP backward (2s/layer × 64 layers via D2H+gloo+H2D); fix: bypass-restore
  `_orig_reduce_scatter_tensor` around `loss.backward()`. See
  `experiments/multinode_32b/test_{w,z,zz,bb,cc}_*.sh` and
  `memory/project_chunked_loop_regression.md` for full test history.
- **32B dedicated-vLLM 2-node (RESTORED 2026-04-23)**: Test CC validated
  `TORCHTUNE_USE_CHUNKED_LOSS=1` at the exact crash config (G=16 fbs=4 max_gen=128,
  6/6 steps clean). Backward 9-13s, memory stable from step 2 (resv=59.13 GiB FLAT,
  tightest rank 0.93 GiB free). Step time ~33s with dedicated vLLM vs ~144s
  training-only — 4.4× speedup from offloading generation to the vLLM node.
- **32B XCCL weight sync end-to-end (VALIDATED 2026-04-23)**: Tests CD + CE confirmed streaming
  XCCL weight sync works for 32B 2-node dedicated-vLLM (3/3 steps clean each). XCCL communicator
  initializes correctly (world=13: 1 training + 3×4 vLLM TP ranks), weight sync completes without OOM.
  - Test CD (707 per-param XCCL calls): 40s/sync at 1.6 GB/s
  - Test CE (66 batched XCCL calls, 512M numel/batch): **38s/sync at 1.7 GB/s** — only 2s improvement
  - **Real bottleneck identified (Test CF, 2026-04-23)**: XCCL **broadcast bandwidth** (1.7 GB/s at
    12 receivers = 35.9s for 61 GiB), NOT AllGather overhead. AllGather per param: <0.12ms for 3B,
    ~0.8ms for 32B (negligible). The earlier "53ms × 707 = 37.5s AllGather floor" was WRONG.
  - **37-40s sync floor** is a BROADCAST BANDWIDTH floor: 61 GiB × (1/1.7 GB/s) = 35.9s regardless of call count.
  - Batching XCCL calls (707→66) saved only ~2s (call overhead reduction), not the data transfer time.
  - BATCHED_AG=1 (batched all_gather_into_tensor) is BROKEN: leaves FSDP2 shard state inconsistent,
    causing checkpoint AllGather to deadlock. Savings were <1% for 32B. Do not use.
  - Teardown fix (Test CF, 2026-04-23): `os._exit(0)` for all ranks after XCCL abort bypasses
    `destroy_process_group()` hang (abort() corrupts oneCCL state). Run A exited cleanly.
  See `experiments/multinode_32b/test_cd_wsync.sh`, `test_ce_wsync_batched.sh`, `test_cf_xccl_1node_3b.sh`.
- **2-hop XCCL weight sync VALIDATED (Test CK, 2026-04-24)**: 3/3 steps clean, 4× sync speedup.
  - Architecture: 2-rank cross PG (training rank 0 → vLLM rank 1, Slingshot) + 12-rank intra PG (vLLM ranks 1-12, XeLink).
    Training uses `broadcast(root=0)` on the 2-rank cross PG (sends 1 copy only). vLLM rank 1 receives via
    cross PG, then broadcasts to ranks 2-12 via intra PG. No send/recv (XCCL send/recv hangs on Aurora).
  - **Sync: 38s → 9.2-9.6s** (bcast=7.7-7.8s at 7.8-8.0 GB/s; 61.02 GiB, 66 batches). 4× speedup.
  - **Step time: 48s/44s/43s** (step 0/1/2) vs 72s with flat broadcast. Converged ~44s steady-state.
  - Memory: tight but clean and STABLE. Between-step reserved: 45.6 → 56.2 → **59.1 GiB (FLAT steps 2-5)**.
    POST-BWD tight rank l0_free=2.61-2.84 GiB; no growth past step 2. 6-step stability run (2026-04-24)
    confirmed: all 6 steps clean, memory converged at 59.13 GiB, step times 40-48s throughout.
  - Bandwidth: 2-rank cross broadcast = 10.1 GB/s; 13-rank flat broadcast = 1.7 GB/s. 6× ratio (XCCL uses
    tree internally, not 12× sequential; confirms bottleneck was broadcast algorithm not pure link count).
  See `experiments/multinode_32b/test_ck_2hop_wsync.sh`, `test_ck_stability.sh`.
- **Test CI: wsync ablation XCCL TP=2 vs TP=4 vs SHM TP=4 (3B, 2026-04-23)**: All 3 runs exit=0.
  - XCCL TP=2 (2 receivers): **6.5 GB/s, 1.4s** per sync (synchronous, in "other" bucket)
  - XCCL TP=4 (4 receivers): **4.0 GB/s, 1.9s** per sync (synchronous, in "other" bucket)
  - SHM  TP=4: **7.2 GB/s write, 0.8s async write** + **0.9s vLLM H2D wait** (page-fault warmup 4.0s first sync only)
  - Key finding: fewer XCCL receivers → proportionally higher bandwidth (TP=2 = 1.625× TP=4)
  - 32B projections at TP=2: **9.4s XCCL** (synchronous) vs **~8s SHM write** (async, hides in gen if gen>8s)
  - SHM is async and write hides in gen; for 32B with gen>15s, SHM overhead ≈ 0; XCCL always costs 9.4s
  - **Tests CJ Final + CJ v7 (2026-04-23)**: 32B XCCL TP=2 on 10+2 single-node COMPLETED — reveals structural limitation.
    - XCCL sync timing CONFIRMED: step 0 = 11.2s (bcast=9.5s at 6.4 GB/s), step 1 = 10.5s (bcast=9.3s at 6.6 GB/s). CI extrapolation was accurate.
    - SHM TP=2 timing CONFIRMED: step 0 waited=49.8s (new SHM block, 1.4 GB/s); step 1 waited=12.1s (reused block, 9.8 GB/s).
    - **STRUCTURAL BLOCKER (both methods crash at step 2)**: oneCCL IPC handle accumulation.
      - Root cause: FSDP2 AllReduce across 10 training ranks caches ~6000 IPC handle mappings (707 params × 9 peers). By end of step 1 backward, **10.85 GiB** of L0 external memory consumed by cached handles, leaving only **5.64 GiB** l0_free. Insufficient for step 2.
      - With threshold=1000 (default): handles evicted → stale VA → banned:1 (Steps CJ v1–v4).
      - With threshold=65536 (fix): no eviction but full 10.85 GiB accumulation → OOM at step 2 (CJ Final, CJ v7).
      - `expandable_segments` setting makes NO DIFFERENCE (tried both True and False — identical memory profiles).
    - **Fix: use 2-node HSDP** — Test CC (6/6 clean at G=16) uses HSDP where within-node AllReduce only covers 5 peers per rank (not 9), dramatically reducing IPC handle memory. Cross-node uses Slingshot, not IPC.
  See `experiments/wsync/test_ci_wsync_ablation.sh`, `test_cj_final.sh`, `test_cj_v7_xccl_expsegs.sh`.

**Next concrete actions** (see Tier 1 below for priority):
1. ~~Submit 3B gene recall production~~ (DONE 2026-04-24): Job 8449766 queued, `experiments/gene_recall/run_3b_gene_recall_production.sh`.
2. **Submit 32B production** — `experiments/multinode_32b/run_32b_2hop_production.sh` ready. 2-hop XCCL confirmed stable (6/6 clean, memory FLAT from step 2). Blocked on debug Q per-user limit (clear when job 8449766 starts).
3. Gemma4-31B training-stability fixes (SFT warm-up, EOS handling, lr warm-up)
4. ~~Run Test CK bandwidth benchmark~~ (DONE 2026-04-24): 2-rank=10.1 GB/s, 13-rank=1.7 GB/s, 6× ratio.
5. ~~Run Test CK 2-hop wsync~~ (DONE 2026-04-24): 3/3 clean, sync 38s→9.4s, step 72s→44s.
6. ~~6-step stability test~~ (DONE 2026-04-24): Memory stable at 59.13 GiB steps 2-5 (no growth). **2-hop XCCL is production-ready.**

**2026-04-24 diagnosis: why 72s/step is not a physics floor**
The flat XCCL broadcast (Tests CD/CE, 38-40s) sends 12 sequential cross-Slingshot copies (one per vLLM TP rank) instead of a tree broadcast. Slingshot 11 per-node injection BW is ~200 GB/s; hardware floor is ~3s (1 Slingshot send at ~25 GB/s + intra-node XeLink at 95 GB/s). The 38s measured is a software/algorithm inefficiency in XCCL's broadcast for cross-node groups, NOT a hardware floor. Fix: 2-hop (implemented in `vllm_weight_sync_worker.py` + recipe).

---

## Current Baselines (validated)

| Model | Config | Step time | Sync waited | Status |
|-------|--------|-----------|-------------|--------|
| Qwen2.5-3B | 10+2 tiles, SHM sync | ~21s | 1.4s | Production-ready |
| Gemma4-31B | 10+2 tiles, SHM sync | ~83s | 12.9–13.5s | Sync stable; training unstable |
| Qwen3-32B | 10+2 tiles, server mode | ~25.6s | HTTP | Stable baseline |
| Qwen3-32B | 2-node HSDP | ~19.4s | — | Near-linear scaling |
| Qwen3-32B | 12 tiles training-only, G=16 fbs=4 max_gen=128 | ~144s | — | 6/6 steps clean 2026-04-23 |
| Qwen3-32B | 2-node dedicated vLLM (TP=4 DP=3 + 12 train tiles), G=16 fbs=4 max_gen=128 | ~33s | — | 6/6 steps clean 2026-04-23 (Test CC) |
| Qwen3-32B | 2-node dedicated vLLM + XCCL weight sync, G=16 fbs=4 max_gen=128 | ~72-76s | 38-40s | 3/3 clean each: Test CD (707 XCCL calls, 40s) & Test CE (66 batched, 38s) 2026-04-23; floor is BROADCAST BANDWIDTH (1.7 GB/s at 12 receivers = 35.9s for 61 GiB); BATCHED_AG=1 broken (checkpoint hang) |
| Qwen3-32B | 2-node dedicated vLLM + **2-hop XCCL** weight sync, G=16 fbs=4 max_gen=128 | **~43s** | **9.1-9.6s** | **Test CK + stability: 6/6 clean 2026-04-24** (exit=0). 2-rank cross PG (Slingshot) + 12-rank intra PG (XeLink). bcast=7.7-7.8s at 7.8-8.0 GB/s. Steps 0-5: 48/44/43/40/43/43s. Memory STABLE: 59.13 GiB from step 2 (no growth steps 3-5, l0_free=2.61-2.84 GiB). 4× sync speedup vs flat broadcast. **PRODUCTION-READY.** |
| Qwen2.5-3B | 1-node XCCL TP=2 (8+2), wsync every step | ~8.7s | 1.4s (sync in "other") | **6.5 GB/s** (Test CI Run A, 2026-04-23); extrapolates to 9.4s for 32B at TP=2 |
| Qwen2.5-3B | 1-node XCCL TP=4 (8+4), wsync every step | ~9.7s | 1.9s (sync in "other") | **4.0 GB/s** (Test CI Run B, 2026-04-23); extrapolates to 15.3s for 32B at TP=4 |
| Qwen2.5-3B | 1-node SHM TP=4 (8+4), wsync every step | ~8.3s | 0.9s waited in gen | **7.2 GB/s write, 0.8s async** (Test CI Run C, 2026-04-23); write hides in gen; 32B: ~8s write (hides if gen>8s) |
| H100 NVL (8×) | torchtune+vLLM 32B | ~15.3s | — | External reference |

---

## Training Run Results (April 15, 2026)

### Qwen2.5-3B gene recall — 1h run (80 steps, job gene_recall_3b_1h)

- Completed 80 steps in 1h (consistent with ~21s/step)
- SHM weight sync: steady-state 2.8s total, 1.4s waited — confirmed hidden behind generation
- Peak memory: 19–23 GiB active, 25–44 GiB reserved (allocator holds freed blocks; not a leak)
- KL well-controlled (kl_coeff=0.3): max spike ~0.08, no runaway divergence
- Reward trajectory: noisy but clearly rising

| Step | Reward | Successes |
|------|--------|-----------|
| 1 | 0.073 | 0% |
| 51 | 0.154 | 12.5% |
| 72 | **0.483** | **50%** |
| 80 | 0.164 | 0% |

**Assessment**: Clear learning signal. Step 72 peak (0.48 reward, 50% success) confirms the
task is learnable with GRPO. No checkpoint was saved (save_every_n_steps not set for this run).
Ready for a longer production run with checkpointing.

### Gemma4-31B gene recall — 1h run (job gene_recall_gemma4_1h)

- `response_lengths=511.0` every step — model never generates EOS in 512 tokens
- `num_stop_tokens=0` every step
- KL explosion: step 4 kl_loss=102,652 (despite clamp in loss.py)
- grad_norm: step 3 = 27,392 → catastrophic divergence

**Assessment**: Training is unstable. Root causes are task-level, not weight sync:
1. Gemma4 hasn't learned the `<genes>...</genes>` output format (no SFT warm-up)
2. With no EOS, all responses are 512-token truncations → reward signal is noisy
3. Policy distribution diverges rapidly from reference → KL explodes

The doc `vllm_weight_sync.md` notes "31B: stable, no OOM" — this refers only to
weight sync stability, not training stability. These are separate concerns.

---

## Issues in `docs/features/vllm_weight_sync.md`

### Issue 1: "Sync fully hidden" is overstated for 3B

The doc says: _"Step time unchanged at ~21s — sync fully hidden."_

The 1.4s waited IS included in the 21s step time — it is not zero. A more precise
statement: sync is **93% hidden** (1.4s of ~20s generation time remains as overhead).
Over 80 steps this adds ~1.8 minutes of wait. The framing matters for setting
throughput expectations in production.

### Issue 2: The 13s "irreducible" wait for 31B is challengeable

The doc claims waited=13s is irreducible. It is only irreducible under the current
serial pipeline:

```
[opt] → [gather 5.5s] → [copy 7s] → [POST→vLLM 6s] → [gen starts]
```

Three concrete paths to reduce it:

**A. Start gather during GRPO forward (1-step weight lag)**
The `full_tensor()` collective must happen after optimizer.step(), but it could start
immediately after the optimizer and complete during the *current* step's forward pass,
feeding the vLLM server for the *next* step. This introduces a 1-step lag in weight
sync — which is algorithmically sound in GRPO (generations are always from the
pre-update policy, as required). Would hide the 5.5s gather entirely.

**B. Single large memmove instead of 832 individual calls**
The current background thread calls `ctypes.memmove` once per parameter (832 calls
for 31B). Python dispatch overhead accounts for the ~8 GB/s rate rather than DDR5
peak (~50 GB/s). A single bulk copy — flatten all gathered tensors into one contiguous
buffer, one memmove call — could plausibly reduce the 7s copy to ~2s.

**C. Accept it and use longer sequences**
The 13s wait is proportional overhead against generation time. For 31B:

| max_gen tokens | gen time (est.) | waited overhead |
|---------------|----------------|-----------------|
| 128 | ~51s | 25% |
| 512 | ~200s | 6.5% |
| 1024 | ~400s | 3.3% |

For meaningful scientific RL (gene recall needs >100 tokens of reasoning), longer
sequences make the 13s wait negligible without any code changes. This is the most
pragmatic path.

### Issue 3: "Why not XCCL" explanation is imprecise

The doc says: _"creating a second XCCL communicator while the training process group
is active causes a fatal error in the Level Zero runtime."_

This is too specific. The observed SIGABRT occurred in **colocate_sleep mode** where
training+vLLM shared the same process and an existing XCCL communicator. In **server
mode**, vLLM runs as a completely separate process. The actual constraint is:
**separate processes cannot form a shared XCCL communicator without a joint distributed
init across all of them**.

The truly unexplored path: launch training + vLLM together under `mpiexec` (all in
the same PMIx job), then create a second process group spanning all ranks. This is
exactly what TRL does with NCCL's `update_named_param`. On XPU with XCCL this may
or may not SIGABRT — the colocate_sleep failure involved an existing XCCL communicator
in the same process, which is a different scenario. Worth a targeted test: launch
a 2-process job where process 0 has an existing XCCL PG and process 1 is a vLLM
worker, and see if `init_process_group` on a new group succeeds.

If it works, this would eliminate SHM entirely — direct HBM→HBM transfer via XeLink
or Slingshot instead of going through CPU DRAM.

### Issue 4: Multi-node weight sync is unaddressed

The doc makes no mention of weight sync in multi-node configurations, but 2-node
32B HSDP is a validated baseline (19.4s/step). `/dev/shm` is **node-local**, so the
current design implicitly assumes vLLM tiles are on the same node as rank 0.

This works **by accident** in current configs (vLLM tiles are always on node 0,
same node as rank 0). But this assumption is undocumented and fragile. Before scaling
multi-node runs, the launcher should:
1. Assert that the vLLM tile indices are on node 0
2. Document that rank 0 (on node 0) writes SHM and vLLM (also on node 0) reads it

For a fully general multi-node + multi-vLLM-server design, a different mechanism
(shared filesystem path, or cross-node mmap via RDMA) would be needed.

### Issue 5: Copy bandwidth claim is misleading

The doc attributes 7-8 GB/s SHM copy speed to "DDR5 ~8 GB/s." DDR5 raw bandwidth
is ~50 GB/s (4-channel). The actual bottleneck is sequential Python dispatch for 832
separate `ctypes.memmove` calls — not the memory bus. This distinction matters because:
- Increasing parallelism (single bulk memmove) would break this ceiling
- Claiming DDR5 as the ceiling incorrectly implies there's no room to improve

### Issue 6: Gemma4 31B training stability issues not in scope

As documented above, the 31B training is unstable independent of weight sync. The doc
should note that "stable" refers to infrastructure (no OOM, no crash), not training
convergence.

---

## Open Architecture Questions

### Q1: Can XCCL broadcast replace SHM?

Unexplored: launch training+vLLM in one MPI job, create a shared process group
spanning all ranks, broadcast from rank 0 to vLLM TP workers. This is the TRL
architecture translated to XPU. Would eliminate SHM entirely and reduce 31B sync
from 13s to whatever XCCL broadcast takes over XeLink (likely 2-4s for 57 GiB).

Test: 2-process XCCL test where one process has an existing PG and we add a second.

### Q2: Expert Parallelism + weight sync interaction

EP (currently at v133, gloo backward AllToAll) is not production-ready. If EP
eventually works, weight sync changes significantly: expert params are EP-sharded
and not covered by the FSDP `full_tensor()` gather. An additional AllGather across
the EP group would be needed before syncing to vLLM. Note this as a dependency for
any EP production plan.

### Q3: vLLM native weight sync API

vLLM RFC [#31848](https://github.com/vllm-project/vllm/issues/31848) proposes
standardized `init_weight_transfer` / `update_weights` / `finalize_weight_update`
endpoints. Our `load_weights_from_shm` extension is the functional equivalent of
their IPC backend. When this API stabilizes, migrate to it — our `/collective_rpc`
injection point is already the correct hook. NCCL backend won't apply (XCCL
constraint), but the SHM/IPC path we built is the XPU analog.

---

## Expert Parallelism Status (as of April 22, 2026)

EP=4/DP=3 for Gemma4 26B-A4B is now at v153. The dispatch algorithm was reformulated
from AllToAll (v18–v136, persistent XCCL SIGSEGV) to **AllGather + ReduceScatter**
(v141+, Mula paper arXiv 2604.00785). Forward unblocked; backward still blocked.

| Component | Status |
|-----------|--------|
| EP dispatch/combine wiring | Fixed (v18) |
| AC non-determinism (topk) | Fixed (v16: argsort stable) |
| Routing imbalance | Fixed (v112: interleaved assignment) |
| FSDP2+EP communicator conflict | Fixed (v40: 1-rank solo FSDP2 for experts) |
| OFI CQ contamination (EPERM) | Fixed (v40: separate fsdp2_ep_pg) |
| AllToAll backward XCCL SIGSEGV | Bypassed by switching to AllGather + ReduceScatter |
| FWD pass (AG+RS, gloo CPU-bounce) | Working — all 180 forward EP ops complete |
| Dedicated `_GLOO_EP_PG` (separate from FSDP2) | Fixed (v152) |
| `GLOO_SOCKET_IFNAME=lo` (bypass CXI NIC) | Fixed (v153) — ranks now reach op #259 |
| **BWD pass desync at op #259** | **Unresolved** — autograd-ordering bug |

### v152 → v153 finding (April 22)

The op #259 RS-BWD deadlock is **not** a transport issue. v153 added per-rank
logging and revealed that ranks within an EP group **execute backward ops in
different orders**. At op #258, ranks 0/2/3/4/6/7/8/10/11 are doing `AG-BWD` while
ranks **1, 5, 9** (the local-index-1 rank in each EP group) are doing `AG-FWD`
recompute of the next layer — they are exactly one op behind their peers.

Gloo matches collectives by call order, so once one rank in a group is out of step,
every subsequent collective in that group references a different layer. After ~19
"lucky" matched ops the mismatch becomes fatal: in v152 the desynced rank never
arrives (gloo TCP timeout); in v153 with loopback transport the rank crashes hard
mid-collective (rank 8 exitcode 1, "Connection closed by peer [127.0.0.1]" cascades
to peers).

Root cause is PyTorch autograd's hook scheduling under AC + `use_reentrant=True`:
the MoE forward hook registration order interacts with autograd's topological sort
in a rank-dependent way, producing different per-rank backward execution orders
even with identical inputs and weights.

Full analysis in `docs/features/moe_integration.md` ("Backward Dispatch Saga (v141–v153)").

### Path forward (revised)

The fix requires decoupling EP collectives from autograd scheduling. Options ranked
by cost/risk:

1. **Move EP dispatch outside autograd** (preferred): rewrite `_token_dispatch` /
   `_token_combine` as side-effecting operations with manual gradient handling.
   Cleanest fix, reusable for any MoE+AC combination.
2. **Disable AC on the 30 MoE layers**: removes the ordering-sensitive recompute.
   Memory cost may force G=2 — worth measuring.
3. **Tag collectives explicitly**: gloo doesn't support tagging well; would need a
   custom barrier per op, killing throughput.

**Benchmark (April 11, EP=1 vs EP=4 at batch=1, before BWD blocker resurfaced):**

| Phase | EP=1 | EP=4 |
|-------|------|------|
| gen | 60.2s | 131s (2.2×) |
| grpo | 31.2s | 74s (2.4×) |
| opt | 1.6s | 39s (24×, CPU AdamW) |
| total | 93s | 246s (2.6×) |

EP=4 was strictly worse at G=1 even before the BWD desync was understood. EP benefit
requires large enough batches for expert GEMM to dominate dispatch latency.

**Recommendation**: EP remains a research effort. Production training continues with
EP=1 (replicated experts) using the proven 12-tile single-node or 2-node HSDP recipes.
Pursue option 1 (EP dispatch outside autograd) as the next concrete experiment if EP
is reprioritized.

---

## Prioritized Next Steps

### Tier 1: Immediate (production training)

**1. Qwen3B gene recall production run**
- Config: kl_coeff=0.3, save_every_n_steps=20, vllm_weight_sync_method=shm
- Run in regular queue (not debug) for >500 steps / multiple epochs
- Monitor: reward plateau, KL spikes, checkpoint integrity
- Expected: continue the 0% → 50% success trajectory from the 1h run

**2. 32B dedicated-vLLM 2-node (Test CC)**
- Apply `TORCHTUNE_USE_CHUNKED_LOSS=1` + XCCL bypass to the 2-node mpiexec launcher
- Note: mpiexec launchers must keep pmix CCL vars; the XCCL bypass is safe with XCCL reduce_scatter on any launcher
- The gloo reduce_scatter patch is still needed for EP mode (EP sub-communicators)
- Config: G=16 fbs=4 max_gen=128 with dedicated vLLM on second node (2 nodes total)
- Expected: ~30-40s/step (vs 144s training-only — vLLM handles generation)
- Script template: `experiments/multinode_32b/test_bb_production.sh` with vllm_url active

**3. Gemma4 31B gene recall — fix training stability first**
- Root cause 1: no SFT warm-up for `<genes>` format → add 5–10 SFT examples as prompt prefix, or cold-start with supervised fine-tuning on 20 examples
- Root cause 2: max_generated_tokens=512 causes all responses to be truncated → reduce to 256, or better: fix EOS generation by adjusting stop tokens config
- Root cause 3: lr too high for warm-up — step 2 grad_norm=3968 suggests initial lr is too aggressive; try linear warm-up over 20 steps
- Only attempt RL training after steps 1-3 stabilize KL

### Tier 2: Infrastructure improvements

**5. Single large memmove for 31B**
- Replace 832 individual memmove calls with one flat-buffer bulk copy
- Expected: reduce 31s copy time from 7s to ~2s (3.5×)
- Estimated effort: 30 lines in `_sync_weights_to_vllm_shm()`

**6. Document multi-node weight sync assumption**
- Add assertion in launcher that vLLM tiles are on node 0 (same node as rank 0)
- Add comment in recipe that SHM is node-local
- Design cross-node sync before scaling vLLM+2-node training

**7. Test XCCL broadcast as SHM replacement**
- 2-process test: process 0 creates XCCL PG, process 1 is vLLM worker, both call init_process_group for a second PG
- If no SIGABRT: broadcast 1 GB tensor and measure bandwidth
- If successful: implement as `vllm_weight_sync_method: xccl` — would eliminate SHM entirely

### Tier 3: Research / deferred

**8. 1-step gather overlap** (reduce 31B waited from 13s to ~7s)
- Restructure training loop to start gather immediately after optimizer step
- Feed gathered weights to vLLM during GRPO forward of the SAME step
- 1-step weight lag is standard in RL — no algorithmic concern

**9. EP backward AllToAll resolution** (deferred)
- v133 (gloo all_reduce) result pending
- If v133 fails: consider abandoning gloo entirely; pre-allocate fixed XPU buffers and use XCCL with static tensor shapes (avoids variable-shape AllToAll entirely)
- Only pursue if EP at large batch sizes shows > 30% throughput improvement

---

## Framework Version Finding (April 21, 2026)

### frameworks/2025.3.1 is NOT broken — `expandable_segments` was the culprit

The previous conclusion that `frameworks/2025.3.1` had a broken XCCL allreduce was **wrong**.
The USM pointer error (`coll_check.cpp:68 ccl_check_usm_pointers: invalid usm pointer type:
unknown`) was caused by `PYTORCH_ALLOC_CONF=expandable_segments:True`, not the framework version.

**Validation matrix (April 21, node x4114c1s7b0n0):**

| Framework | `expandable_segments` | Result |
|-----------|----------------------|--------|
| 2025.3.1 (torch 2.10, Python 3.12) | unset | **PASS** — `allreduce OK, sum=2.0` |
| 2025.3.1 (torch 2.10, Python 3.12) | `True` | **FAIL** — `invalid usm pointer type: unknown` |
| 2025.3.1, full GRPO 2-tile | unset | **PASS** — 26.5s/step, 2 steps, metrics healthy |

**Root cause**: In torch 2.8 (frameworks/2025.2.0), the XPU allocator ignored all
`PYTORCH_ALLOC_CONF` settings — `expandable_segments:True` was a no-op. In torch 2.10
(frameworks/2025.3.1), the allocator config is actually honored. `expandable_segments`
changes how memory is allocated (likely `zeVirtualMemReserve` + `zeVirtualMemMap` instead
of `zeMemAllocDevice`). The resulting virtual memory pointers are reported as "unknown" USM
type by Level Zero's `zeMemGetAllocProperties`, which oneCCL's `ccl_check_usm_pointers`
rejects during collective operations.

**Implications:**
1. Can use `frameworks/2025.3.1` (torch 2.10, Python 3.12) for all workloads — just
   `unset PYTORCH_ALLOC_CONF` or never set `expandable_segments:True`
2. The same error appears in the vLLM TP>1 context (documented in Phase 12 bug fix) —
   same root cause, same fix
3. `expandable_segments` would be valuable for reducing memory fragmentation (could help
   with G=16 OOM). This is a legitimate Intel bug: oneCCL should accept pointers from
   PyTorch's own XPU allocator. Worth filing with ALCF/Intel.

### `expandable_segments` IS implemented and works on XPU — but CCL rejects the pointers

**Validated (April 21, node x4706c3s2b0n0):** Single-process allocation stress test
(`_test_expandable_segments.py`) confirms `expandable_segments` genuinely reduces
fragmentation on XPU in torch 2.10:

| Alloc size | Reserved (without ES) | Reserved (with ES) | Savings |
|-----------|----------------------|-------------------|---------|
| 32MB | 117.4MB | 83.9MB | 29% |
| 64MB | 184.5MB | 83.9MB | 55% |
| 128MB | 318.8MB | 146.8MB | 54% |
| 256MB | 587.2MB | 272.6MB | 54% |
| 512MB | 1124.1MB | 545.3MB | 51% |

Without ES, each larger allocation roughly doubles the reserved gap (buddy allocator
fragmentation). With ES, segments expand in-place — the gap stays tight. At production
scale (multi-GB FSDP shards), this could recover **several GiB** of fragmented memory
per tile. Note: G=16 OOM was caused by the per-chunk fwd+bwd loop regression (commit acdc7c9f),
not fragmentation — now fixed with `TORCHTUNE_USE_CHUNKED_LOSS=1`. G=16 fbs=4 max_gen=128
runs 6/6 steps clean on a single node with the fix applied.

**The problem is strictly in oneCCL's USM validation.** `expandable_segments` uses
`zeVirtualMemReserve` + `zeVirtualMemMap` which produces pointers that `zeMemGetAllocProperties`
reports as "unknown" USM type. oneCCL's `ccl_check_usm_pointers` (`libccl.so.1.0`) rejects
these with no env var bypass available. **This should be filed as an Intel/oneCCL bug** —
the allocator and the collective library are both Intel code, and they should be compatible.

### frameworks/2025.3.1 validated at production scale

**12-tile full-node GRPO (April 21, node x4706c3s2b0n0):**

| Step | Total | Gen | GRPO | Memory (reserved) |
|------|-------|-----|------|------------------|
| 0 | 47.4s | 41.5s | 4.2s | 13.1 GiB |
| 1 | 39.8s | 37.5s | 2.3s | 13.0 GiB |

All 12 ranks healthy. Loss, rewards, grad_norm, KL all normal. No USM errors, no OOM.
frameworks/2025.3.1 is production-ready with `unset PYTORCH_ALLOC_CONF`.

---

## Framework Position Summary

| Framework | Platform | Best step time (32B, G=4, 128tok) | Notes |
|-----------|----------|----------------------------------|-------|
| torchtune + SHM vLLM | Aurora XPU | **18.1s** | Validated; multi-node 19.4s |
| TRL + vLLM | H100 NVL | 10.9s | Best A100-class result |
| torchtune + vLLM | H100 NVL | 15.3s | Same code, H100 18% faster |
| verl + vLLM | A100 40GB | 21.6s | Weight sync overhead |

Aurora with torchtune is **1.18× slower than H100 NVL** on the same task with the same
code (18.1s vs 15.3s). This is expected given H100's compute advantage. The framework
choice (torchtune over TRL/verl) contributes 32-35% throughput advantage on both
platforms — validating the original TRL-over-torchtune rationale.
