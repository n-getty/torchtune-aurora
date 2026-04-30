# Project Status — Aurora RL (torchtune XPU)

Last updated: 2026-04-30 (Qwen3-30B-A3B EP=4/DP=6 v8 series: first `loss=` line at G=1/fbs=1/NSTEPS=1 on 3-node hold; v8m_b + v8n diagnose unified architectural blocker — v59 `reduce_grads=False` keeps ~18.5 GiB FSDP2 unsharded grads alive across BOTH chunk and step boundaries, blocking G≥2 and NSTEPS≥2 alike)

This document synthesizes the current state of the vLLM weight sync implementation,
active training runs, open issues, and prioritized next steps. It is a living companion
to `docs/features/vllm_weight_sync.md`, `docs/features/moe_integration.md`, and `docs/experiments/aurora_rl_baselines.md`.

## Where we are (one-page stock-take)

- **Production-ready paths**:
  - Qwen2.5-3B: 10+2 tiles, SHM sync, 21s/step. **130 steps clean** with `usm_caching_alloc.so` (job 8449766).
  - Qwen3-30B-A3B (MoE): 10+2 tiles, SHM sync + GPU transpose + GPU fuse. **G=8 is production config**: 54.8s/step, 9.2 tok/s, 3/3 steps clean (2026-04-27). G=4 OOMs at step 1 (allocator fragmentation). MoE weight sync: 531 fused params, 13s vLLM reload, 3.3s gather. See `docs/features/moe_integration.md`.
  - Qwen3-32B (2-node): **2-node dedicated vLLM + 2-hop gloo cross-PG**, ~67s/step. **20/20 steps clean, exit=0** (Run 8, 2026-04-25). CXI RDMA leak eliminated. R2-R10 FLAT. **This is the recommended 2-node 32B path for runs >30 steps.** For short runs (<30 steps), XCCL cross-PG is faster (~43s/step, 24/24 clean).
  - Qwen3-32B (3-node): **3-node 24-way pure FSDP** (1 vLLM node + 2 training nodes). Gloo cross + XCCL intra: **53.5s avg, 20/20 clean, 10/10 syncs** (v14). XCCL cross + XCCL intra: **53.9s avg, 5/5 clean** (v16b). DP>1 (3 replicas, parallel broadcast): **59.9s avg, 5/5 clean** (v18) — benefits at high gen load. **Pinned CPU buffer** (`TORCHTUNE_PINNED_CPU_BUF=1`): gather 31s → 3.7s (8.5× speedup), **~41s avg at G=16, 5/5 clean** (Test A). **G=32 + pinned buffer**: **~53s avg, 5/5 clean** (Test B), 1.54× per-sample throughput vs G=16. **Production envelope mapped (2026-04-29)**: G=32/max_gen=128 is best throughput (0.60 samples/s); G=32/max_gen=192 is marginal (1.50 GiB free); G=48+ blocked by **2-chunk rule** (CCL external memory explosion from 1.8→13-15 GiB when G/fbs > 2). G=64 also hits XPU kernel indexing bug at batch=64. XCCL intra crash (v11) **RESOLVED** — stale L0. DP>1 base_rank bug **FIXED** (bug #18). WSYNC_CROSS_METHOD passthrough **FIXED** (bug #19).
  - Gemma4-26B-A4B: EP=1, 24s/step.
  - **BioReason-Pro 4B (multimodal RL, NEW 2026-04-29)**: two paths now working.
    - **Single-node (run 41/42 baseline)**: 11+0 FSDP1 SHARD_GRAD_OP, ~43-45s/step, 20/20 clean. Fix combination = drop `_multimodal` chunked-loss gate + persistent wsync chunk buffer + G=4/fbs=4. See `docs/reports/bioreason_4b_status_20260429.md`.
    - **2-node asymmetric server mode (Phase 2, runs 49/50, job 8457145)**: 11 train ranks on TRAIN_NODE + 12 vLLM HTTP servers on VLLM_NODE (DP=12, ports 8001-8012). Multimodal pipeline (ESM3 + GO encoder + protein projection + Qwen3 embed_tokens) stays on train side; final `[B*G, T, hidden]` embed shipped over HTTP via base64-encoded `torch.save(bf16)` → vLLM `/v1/completions {prompt_embeds: ...}`. Weight sync via shared-FS `/lus/flare/.../weight_update.raw` → `/collective_rpc load_weights_from_raw` (399 backbone-only params, ~6-8s save @ 1.0-1.3 GB/s + 8.5-10.7s vLLM load). **Run 50: 5/5 steps clean, 4 consecutive successful syncs, KL evolves (0.0016-0.0027)**. Step time ~88-95s with `other=40-47s` blocking on wsync wait — async overlap not yet realized in server mode (optimization target before capacity 4h run). See `docs/reports/bioreason_4b_phase2_20260429.md`.
    - **Run 51 (2026-04-29) parallel POST fan-out**: ThreadPoolExecutor over 12 vLLM tiles dropped step time 95.8s → 58s (39% reduction). other=47s → 5s. 5/5 clean, KL 0.002-0.006, memory FLAT.
    - **wsync correctness bug (FIXED 2026-04-30)**: `GRPOBioReasonDistributedXPU.train()` override never called `_sync_weights_to_vllm`. vLLM rollouts in runs 47/50/51 used SFT-initial weights for the entire run — training optimized the policy normally, rollouts never saw the updates. Fix: mirror base recipe lines 3380-3392 after `optimizer.step()`. Validated on hold 8460176, G=4 fbs=4 max_gen=1024 NSTEPS=5: **5/5 clean, weight versions 1→5, 399 params/step, memory FLAT 52.87 GiB steps 2-5, +7s/step over baseline (43→50s)**. See `docs/reports/bioreason_wsync_fix_20260430.md` and `memory/bugs/project_bioreason_train_missing_wsync.md`.
    - **G-bisect blocker (2026-04-30)**: G>4 with parallel HTTP fan-out hits a banned:1 PDE Segfault whose crash step is inversely correlated with G — G=4 ≥15 steps clean; G=8 step 13; G=12 step 3; G=16 step 2. Same root-cause class as `memory/bugs/project_ccl_ipc_handle_cache.md` (cumulative IPC-handle accumulation in CCL/L0 that scales with G). Falsified hypotheses: HTTP fan-out concurrency cap (G=16 cap=4 still crashes) and chunk count (G=16 fbs=4 still crashes). Side-effect: any banned:1 crash makes the hold's L0 device state unrecoverable (`torch.xpu.device_count()=0` after pkill+clean_tiles); bisects require fresh holds. See `memory/feedback_banned1_destroys_xpu.md`.
    - **G=8 unblocked by IPEX `varlen_attention` (NEW 2026-04-30)**: Set `TORCHTUNE_USE_IPEX_VARLEN=1` to route causal-only SDPA calls through `intel_extension_for_pytorch.llm.functional.varlen_attention` with a persistent output buffer (added in `torchtune/modules/attention_utils.py`). Validated on hold 8460388: G=8 NSTEPS=15 **15/15 clean exit=0** (vs baseline crash at step 13); steady-state reserved memory **5 GiB lower** (57.21 vs 62.28 GiB); GRPO step ~19% faster at G=4. Bit-exact vs PyTorch SDPA on XPU. **Production envelope: G=8 fbs=8 max_gen=1024 (~2× rollout throughput vs G=4).** G=12+ ceiling not yet tested. Worth trying on other XPU recipes (32B dense, Qwen3-30B-A3B MoE, gene_recall) — same micro-bench wins (21% kernel speedup + 64% lower per-call peak transient memory) should apply. Falls back to PyTorch SDPA when conditions don't apply (mask present, dropout > 0, non-causal, non-XPU). See `docs/reports/bioreason_ipex_varlen_20260430.md` and `memory/project_bioreason_ipex_varlen_20260430.md`.
    - **`force_math_sdpa` is a no-op on XPU (NEW 2026-04-30)**: The config flag in `grpo_full_finetune_distributed_xpu.py:307-311` calls `torch.backends.cuda.enable_flash_sdp(False)` which only affects the CUDA dispatcher, not the XPU SDPA path. Validated micro-bench: identical timing AND identical peak memory regardless of the flag. Don't vary it expecting behavioral change on XPU. See `memory/feedback_force_math_sdpa_xpu_noop.md`.
- **Async GRPO**: **PAUSED 2026-04-30**, infrastructure complete and validated. Resume after sync production runs and other baselines are hardened. Status:
  - Phase 1 (server-mode, RolloutProducer overlap) validated on Qwen2.5-3B; 50-step k=1 convergence run clean.
  - Phase 2 dedicated-rank (Steps 1+3+4): Qwen2.5-3B 1-node 20/20 clean, ratios in [0.9998, 1.0014]. Step 5 32B 2-node sync smoke validated (5/5 clean, ~241s/step on baseline pair: gen 77s + grpo 110s + wsync 42s).
  - **A2 architectural async (vLLM bg recv + rank 0 bg send) — implemented and validated 5/5 clean (job 8461452, hold pair x4702c6s4b0n0+x4706c1s7b0n0), but NET NEGATIVE on Aurora**. Sender-side broadcast deferred (rank 0 main thread freed: 35s pack+d2h, broadcast deferred to bg thread); receiver-side bg recv overlaps with `vllm.generate`. Ratios=1.0000 (atomic chunk-staged apply preserves correctness); engine lock contention zero (`lock_wait=0.00s`). **The bottleneck is hsn0 bandwidth contention**: bg gloo broadcast competes with FSDP XCCL collectives → bcast bandwidth collapses 3.21 GB/s → 0.27-0.29 GB/s, wall-clock ~300s/step (~25% slower than baseline). Code gated behind `TORCHTUNE_VLLM_BG_WSYNC=1` + `TORCHTUNE_BG_WSYNC_SEND=1` (defaults off, no impact on production sync path). To resume: either throttle bg send to backward-only windows (avoid concurrent FSDP AG/RS), or pivot to algorithm-level async (k≥2 with `RolloutProducer` hiding *generation* — Phase 2 Step 2 — where gloo gen_pg doesn't fight FSDP).
  - This is the architectural baseline for 32B/MoE async runs when revisited; gen=77s vs grpo=110s = ~30% headroom for true async overlap when `async_generation.enabled=true`.
  - See `docs/features/async_implementation.md`, `memory/project_phase2_32b_2node_validated.md`, and `memory/project_phase2_a2_paused.md`.
- **Weight sync**: 2-hop with **gloo cross-PG** + XCCL intra is the production method for 32B runs >30 steps (47s sync at 1.3 GB/s; eliminates CXI RDMA leak). For short runs, XCCL cross-PG is faster (9.1s sync at 7.9 GB/s but leaks ~9 MiB/step → crash ~step 30).
  SHM remains viable for 3B (1.4s sync, fully hidden behind generation).
- **Framework**: `frameworks/2025.3.1` is production-ready with `unset PYTORCH_ALLOC_CONF`.
  `expandable_segments:True` is an Intel oneCCL bug (USM pointer rejection) — file
  upstream; works but unusable with collectives.
- **Expert Parallelism (EP=4/DP=3 for Gemma4 26B-A4B)**: not production-ready.
  Forward works end-to-end; backward fails deterministically at op #259. v153 ruled
  out the CXI-NIC-contamination hypothesis and exposed a per-rank autograd ordering
  bug: ranks within an EP group execute backward ops in different orders, causing
  gloo collective mismatch. See `docs/features/moe_integration.md` for the v141–v153 saga.
- **Expert Parallelism (EP=4/DP=6 for Qwen3-30B-A3B, UPDATED 2026-04-30)**: **first `loss=` line reached** (v8i, G=1/fbs=1/NSTEPS=1 on 3-node hold). v8g per-EP-OP MEMPROBE refuted the IPC-handle hypothesis: external CCL/IPC stays FLAT at ~3.4 GiB, all ~45 GiB growth is `torch_alloc` from saved activations — `banned:1 PDE` was activation OOM all along. **Both G≥2 (v8m_b) and NSTEPS≥2 (v8n) blocked by the same architectural issue**: v59's `reduce_grads=False` (`grpo_full_finetune_distributed_xpu.py:1440-1464`) keeps the full ~18.5 GiB FSDP2 unsharded grad pool resident across both chunk and step boundaries; `optimizer.zero_grad(set_to_none=True)` doesn't reach FSDP2's internal grad buffers. Halving `max_gen` 32→16 only reduces per-chunk activations by <1 GiB — fixed buffers (KV cache, prompts, ref_logprobs) dominate. **Unblocking requires a code change**: explicit `fully_shard`-aware reduce_scatter between chunks AND after the optimizer step. Production RL continues on EP=1 (G=8, 54.8s/step). See "Qwen3-30B-A3B Expert Parallelism Status (2026-04-30, v8 series)" section below; full writeup `docs/reports/qwen3_ep_v8_3node_20260430.md`.
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
  6/6 steps clean). With 2-hop XCCL weight sync: **24/24 steps clean** (gc:0.6,
  job 8450367, 2026-04-24). Memory in steady state — two plateaus (59.13 GiB steps 3-6,
  62.04 GiB steps 7-24), OFI MR stable at 1.56 GiB. Step time ~43s.
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
  - Memory: tight but clean and STABLE. Between-step reserved: 45.6 → 56.2 → **59.1 GiB (FLAT steps 2-6)**,
    then 62.04 GiB from step 7 (one-time +2.91 GiB jump, FLAT for 17 consecutive steps through step 24).
    POST-BWD tight rank l0_free=0.39 GiB; stable. **24-step stress test** (2026-04-24, job 8450367):
    24/24 clean, walltime-limited. OFI MR=1.56 GiB stable throughout. Genuine steady state.
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

- **MoE P0/P1 experiments (2026-04-27)**: Config tuning + torch.compile for Qwen3-30B-A3B.
  - **G=8 (winner)**: batch=1, grpo_samples=8, fbs=8. 3/3 steps, 54.8s/step warm, 9.2 tok/s. 2× RL samples/step, 1.8× throughput vs G=4. Memory stable: peak_resv=62.43 GiB (0.41 GiB free on worst rank), no growth after step 1.
  - **G=4**: 1/3 steps. OOM step 1 backward — allocator fragmentation (reserved=43.35 GiB, allocated=30.40, gap=12.95 GiB → can't satisfy backward allocation from fragmented blocks).
  - **batch=2**: 0/3 steps. OOM step 0 backward — 8 sequences (2×4) exhaust memory before backward starts.
  - **torch.compile**: SYCL kernel compilation impractical on XPU. Full model: 500+ kernels, 25+ min, didn't finish in 1h. Attention-only (with `@torch.compiler.disable` on expert forward): 144 kernels compiled before walltime expired. Root cause: `icpx` SYCL C++ compilation is ~10× slower than Triton PTX generation. Not viable without AOT compilation or kernel caching.
  - **FSDP communication dominance confirmed**: GRPO fwd+bwd = 42.8-43.5s for 8 seqs, ~85% is AllGather/ReduceScatter for 30B total params. EP remains the only path to reduce communication.
  - Report: `docs/reports/moe_p0p1_experiments_20260427.md`

**Next concrete actions** (see Tier 1 below for priority):
1. ~~Submit 3B gene recall production~~ (DONE 2026-04-24): Job 8449766, 130 steps clean with `usm_caching_alloc.so`. Noisy reward (peak 43.75% success at steps 44/84). 3 checkpoints saved.
2. ~~Submit 32B production~~ (DONE 2026-04-24): Job 8450367, **24/24 steps clean** with gc:0.6 + 2-hop XCCL. Memory in steady state. Terminated by PBS walltime, not crash.
3. ~~Static XCCL buffer fix + interval=2~~ (DONE 2026-04-25): Static buffers eliminate VA churn leak on 10/12 ranks. `weight_sync_interval=2` reduces CXI residual leak from 9→5 MiB/step. 14 steps clean, l0_free=0.09 GiB at step 13 (crash est ~step 31).
4. Gemma4-31B training-stability fixes (SFT warm-up, EOS handling, lr warm-up)
5. ~~Run Test CK bandwidth benchmark~~ (DONE 2026-04-24): 2-rank=10.1 GB/s, 13-rank=1.7 GB/s, 6× ratio.
6. ~~Run Test CK 2-hop wsync~~ (DONE 2026-04-24): 3/3 clean, sync 38s→9.4s, step 72s→44s.
7. ~~24-step stress test~~ (DONE 2026-04-24): 24/24 clean. Memory: 59.13→62.04 GiB (one-time jump step 7, then FLAT 17 steps). OFI MR=1.56 GiB stable. **2-hop XCCL is production-ready.**
8. ~~Gloo cross-PG fix~~ (DONE 2026-04-25): Replace XCCL cross-node PG with gloo (TCP/Slingshot). **20/20 steps clean, exit=0.** CXI RDMA leak eliminated on R2-R10 (external FLAT at 1.50 GiB). R0/R1 residual ~5 MiB/step (crash est ~step 55). Broadcast: 47s at 1.3 GB/s (vs 9.2s at 8 GB/s with XCCL). Step time: ~67s (vs ~44s). Critical: `GLOO_SOCKET_IFNAME=hsn0` required.
9. Production throughput optimization: Gloo cross-PG adds ~23s/step overhead from slower TCP broadcast. Options: larger batch sizes, gloo buffer tuning, or accept 67s/step for memory stability.
10. ~~PG reset for XCCL~~ (FAILED 2026-04-27): Run 19c tested periodic PG reset (every 10 steps) + streaming gather + userfaultfd + gc:0.99. PG reset works mechanically (gen=10/20 logged, no deadlocks) but does NOT reduce external memory — CXI MR cache entries persist after PG destruction. Crashed step 29 (same as Run 18).
11. **ROOT CAUSE IDENTIFIED (2026-04-28):** Run 21 (Gloo TCP + sender rotation pool=9) eliminated ALL weight-sync CXI traffic. Still crashed step 29. Non-sender ranks (R0, R1) and vLLM rank (R11) — which never touched any weight sync CXI path — showed identical external=3.52-3.57 GiB after contraction. **Root cause is FSDP AllGather/ReduceScatter CXI MR entries becoming stale after caching allocator contraction, NOT weight sync.** No weight-sync-side fix can resolve this. Must prevent the contraction itself (reduce per-tile memory) or wait for Intel SHS 13.1.0. See `docs/reports/cxi_mr_step28_crash_investigation_20260428.md`.
12. **3-node 24-way FSDP (2026-04-28):** v11 crash **RESOLVED** — stale L0 device state from v9 crash, not XCCL bug. v12 (XCCL intra, fresh L0): **5/5 clean, 56.1s/step avg**. v13 (gloo intra fallback): **5/5 clean, 72.1s/step avg**. XCCL intra 2.4x faster (2.2 vs 0.9 GB/s). Gloo intra adds ~40s to sync-step gen time. Memory FLAT: external 1.93-2.21 GiB, torch_resv=42.29 GiB. DP>1 per-replica PGs untested.

**2026-04-24 diagnosis: why 72s/step is not a physics floor**
The flat XCCL broadcast (Tests CD/CE, 38-40s) sends 12 sequential cross-Slingshot copies (one per vLLM TP rank) instead of a tree broadcast. Slingshot 11 per-node injection BW is ~200 GB/s; hardware floor is ~3s (1 Slingshot send at ~25 GB/s + intra-node XeLink at 95 GB/s). The 38s measured is a software/algorithm inefficiency in XCCL's broadcast for cross-node groups, NOT a hardware floor. Fix: 2-hop (implemented in `vllm_weight_sync_worker.py` + recipe).

---

## Current Baselines (validated)

| Model | Config | Step time | Sync waited | Status |
|-------|--------|-----------|-------------|--------|
| Qwen2.5-3B | 10+2 tiles, SHM sync | ~21s | 1.4s | Production-ready; 130 steps clean with usm_caching_alloc (job 8449766) |
| Qwen3-30B-A3B (MoE) | 10+2 tiles, SHM sync, **G=8** fbs=8 | **~54.8s** | ~3.3s | **3/3 steps clean** (2026-04-27 P0/P1). 9.2 tok/s (1.8× vs G=4). 531 fused params, 13s vLLM reload. Memory stable (peak 62.43 GiB, 0.41 GiB free). XCCL blocked (UR:40 on 10 FSDP ranks). |
| Gemma4-31B | 10+2 tiles, SHM sync | ~83s | 12.9–13.5s | Sync stable; training unstable |
| Qwen3-32B | 10+2 tiles, server mode | ~25.6s | HTTP | Stable baseline |
| Qwen3-32B | 2-node HSDP | ~19.4s | — | Near-linear scaling |
| Qwen3-32B | 12 tiles training-only, G=16 fbs=4 max_gen=128 | ~144s | — | 6/6 steps clean 2026-04-23 |
| Qwen3-32B | 2-node dedicated vLLM (TP=4 DP=3 + 12 train tiles), G=16 fbs=4 max_gen=128 | ~33s | — | 6/6 steps clean 2026-04-23 (Test CC); no weight sync |
| Qwen3-32B | 2-node dedicated vLLM + XCCL weight sync, G=16 fbs=4 max_gen=128 | ~72-76s | 38-40s | 3/3 clean each: Test CD (707 XCCL calls, 40s) & Test CE (66 batched, 38s) 2026-04-23; floor is BROADCAST BANDWIDTH (1.7 GB/s at 12 receivers = 35.9s for 61 GiB); BATCHED_AG=1 broken (checkpoint hang) |
| Qwen3-32B | 2-node dedicated vLLM + **2-hop XCCL** weight sync, G=16 fbs=4 max_gen=128 | **~43s** | **9.1s** | **24/24 steps clean** (job 8450367, walltime-limited). gc:0.6. Memory steady state: 59.13→62.04 GiB (one-time jump step 7, FLAT 17 steps). OFI MR=1.56 GiB stable. XCCL sync 9.1s at 7.9 GB/s. **PRODUCTION-READY.** |
| Qwen3-32B | 2-node dedicated vLLM + 2-hop XCCL + **interval=2**, G=16 fbs=4 max_gen=128 | **~38s** | 9.1s (every 2 steps) | **14/14 steps clean** (Run 6, 2026-04-25). External memory FLAT post-expansion (R0: 1.31→1.34, R1: 1.82→1.85 over steps 7-13). l0_free: 0.13→0.09 GiB (5 MiB/step vs 9 MiB/step with interval=1). ~30 steps before crash est. Walltime-limited. |
| Qwen3-32B | 2-node dedicated vLLM + 2-hop **gloo cross-PG** + XCCL intra, G=16 fbs=4 max_gen=128 | **~67s** | 47s bcast + 32s wait | **20/20 steps clean, exit=0** (Run 8, 2026-04-25). **CXI RDMA leak ELIMINATED** on R2-R10 (external FLAT at 1.50 GiB). R0/R1: 5 MiB/step residual (crash est ~step 55). torch_resv=62.04 FLAT. Gloo TCP at 1.3 GB/s (vs XCCL 8 GB/s). `GLOO_SOCKET_IFNAME=hsn0` critical. |
| Qwen3-32B | 2-node dedicated vLLM + 2-hop XCCL + **PG reset** + streaming gather, G=16 fbs=4 max_gen=128 | **~67s** | 15s bcast + 29s gather | **29/40 (crashed step 29)** (Run 19c, 2026-04-27). PG reset (gen=10/20) works but doesn't reduce CXI MR cache. Contraction at step 28 (torch_resv 62→46 GiB, fwd 45-76s). Survived step 28, crashed step 29 banned:1 (identical to Run 18). |
| Qwen3-32B | 2-node dedicated vLLM + **sender rotation pool=9** + XCCL broadcast, G=16 fbs=4 max_gen=128 | **~65s** | 47s bcast | **28/40 (crashed step 28)** (Run 20, 2026-04-28). Rotation works perfectly (R2→R3→...→R10→R2). R0 external FLAT at 1.03 GiB. Contraction from training memory pressure, not weight sync. |
| Qwen3-32B | 2-node dedicated vLLM + **sender rotation pool=9 + Gloo TCP** (zero CXI wsync), G=16 fbs=4 max_gen=128 | **~80s** | 47s bcast | **29/40 (crashed step 29)** (Run 21, 2026-04-28). **PROOF: root cause is FSDP collectives.** ALL ranks (incl. non-sender R0/R1 and vLLM R11) show identical external=3.52-3.57 GiB after contraction. Zero CXI wsync traffic, still crashes. |
| Qwen3-32B | **3-node 24-way** XCCL intra, G=16 fbs=16 max_gen=128 | **56.1s avg** | ~0.3s (deferred) | **v12 5/5 clean, exit=0** (2026-04-28). v11 crash was stale L0 — fresh job works. XCCL intra 2.2 GB/s. |
| Qwen3-32B | **3-node 24-way** gloo intra, G=16 fbs=16 max_gen=128 | **72.1s avg** | ~0.3s (deferred) | **v13 5/5 clean, exit=0** (2026-04-28). Gloo intra fallback 0.9 GB/s. +28.5% vs XCCL intra. |
| Qwen3-32B | **3-node 24-way + pinned CPU buffer**, G=16 fbs=16 max_gen=128 | **~41s avg** | ~0.3s (deferred) | **Test A 5/5 clean, exit=0** (2026-04-28). Gather 31s→3.7s (8.5×). `TORCHTUNE_PINNED_CPU_BUF=1`. |
| Qwen3-32B | **3-node 24-way + pinned buf + G=32**, fbs=16 max_gen=128 | **~53s avg** | ~0.3s (deferred) | **Test B 5/5 clean, exit=0** (2026-04-28). 1.54× per-sample throughput. Memory tight (l0_free=4.9 GiB step 4) but stable. |
| Qwen3-32B | 3-node 24-way + pinned buf, G=16, fbs=16, **max_gen=512** | **~82s** | ~0.3s (deferred) | **Test D 3/3 clean** (2026-04-29). 0.01 GiB free POST-BWD (absolute limit). resp_len=511 (all seqs hit cap). |
| Qwen3-32B | 3-node 24-way + pinned buf, **G=64**, fbs=16, max_gen=128 | — | — | **Test E FAILED** (2026-04-29). XPU kernel indexing bug at batch=64 in `batched_logits_to_logprobs`. Not OOM. |
| Qwen3-32B | 3-node 24-way + pinned buf, **G=32, max_gen=256** | — | — | **Test G OOM step 1** (2026-04-29). CCL external memory explosion 1.85→13-20 GiB (3 forward chunks). |
| Qwen3-32B | 3-node 24-way + pinned buf, **G=32, max_gen=192** | **~72s** | ~0.3s (deferred) | **Test G2 3/3 clean** (2026-04-29). 1.50 GiB free POST-BWD. Marginal but viable. resp_len=191. |
| Qwen3-32B | 3-node 24-way + pinned buf, **G=48**, fbs=16, max_gen=128 | — | — | **Test H HUNG step 1** (2026-04-29). CCL external explosion to 15.26 GiB (3 chunks). 0.01 GiB free. **2-chunk rule confirmed.** |
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

### Qwen2.5-3B gene recall — 7h production run (130 steps, job 8449766, 2026-04-24)

- **130 steps clean** with `usm_caching_alloc.so` pluggable allocator. Terminated by PBS walltime (SIGTERM), not crash.
- First 2 runs (without caching allocator) crashed at steps 7-10 with `banned: 1` GPU page faults (expected — validates the need for caching allocator)
- MEMPROBE blind (peak_memory shows 0.0 with pluggable allocator)
- KL well-controlled: approx_policy_kl = 0.0003-0.0007 range throughout
- 3 checkpoints saved: epoch_0, epoch_1, epoch_2 (~18 GiB each in `outputs/gene_recall_production/ref/`)

| Step | Reward | Successes | Notes |
|------|--------|-----------|-------|
| 1 | 0.076 | 0% | |
| 17 | 0.369 | 37.5% | Best early |
| 44 | **0.449** | **43.75%** | Peak |
| 84 | 0.419 | 43.75% | Second peak |
| 100 | 0.000 | 0% | |
| 130 | 0.030 | 0% | |

**Assessment**: Noisy reward trajectory — 30 out of 130 steps had non-zero successes, but no
clear monotonic convergence. Best success rate is 43.75% (matching the earlier 1h run's 50%).
The lack of sustained improvement over 130 steps suggests either (a) the learning rate or
G=16 is insufficient for stable convergence on this task, or (b) the reward signal is too
sparse (binary gene match) for GRPO to make steady progress. Next steps: try larger G,
curriculum (easier genes first), or supervised warm-up before RL.

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

## 3-Node 24-Way FSDP for Qwen3-32B (2026-04-28)

### Architecture

```
Node 0 (vLLM):    4 tiles → TP=4, DP=1 vLLM server
Node 1 (Training): 12 tiles → 24-way pure FSDP (dp_replicate=1)
Node 2 (Training): 12 tiles →
```

Weight sync uses 2-hop gloo cross-PG: training rank 0 → vLLM TP-0 (gloo TCP over
Slingshot, ~1.3 GB/s) → all TP workers (XCCL intra-PG over XeLink). Deferred
broadcast: weights gathered at end of step N, broadcast during step N+1's generation
phase (interval=2 means broadcast every other step).

### Optimization stack (cumulative 37% improvement)

| Optimization | Before | After | Savings | Mechanism |
|-------------|--------|-------|---------|-----------|
| policy_fwd elimination | 12.3s | 0.0s | 12.3s | Single-epoch GRPO: `old_logprobs = pi_logprobs.detach()` in train fwd (skip redundant no-grad fwd) |
| FBS=16 | ref_fwd 11.9s | 5.3s | 6.6s | 1 FSDP AllGather round instead of 4 (CLI override `forward_batch_size=16`) |
| Gloo deferred weight sync | ~9s sync overhead | ~0.3s on even steps | ~8.7s avg | Deferred broadcast hides behind gen; interval=2 halves sync frequency |
| **Total** | **52.1s** | **32.6s** | **19.5s (37%)** | |

### v11 step-time breakdown (step 1, DP=1)

| Component | Time | Notes |
|-----------|------|-------|
| gen | 12.4s | vllm=7.9s + ref_fwd=5.3s (policy_fwd=0s, skipped) |
| grpo | 19.7s | fwd=5.6s + bwd=14.1s |
| clip+opt | 0.2s | |
| other | 0.3s | deferred wsync on even steps only (interval=2) |
| **total** | **32.6s** | |

### Run history

| Run | DP | Steps | Result |
|-----|-----|-------|--------|
| v9 | 3 (DP=3 vLLM, FSDP dp_replicate=1) | 2/10 | DP>1 intra-PG deadlock after step 1 — shared intra-PG (size=world_size-1) deadlocked because collective_rpc dispatches per-replica |
| v10 | 1 | 0/5 | vLLM UR_RESULT_ERROR_OUT_OF_RESOURCES — stale L0 device contexts from v9 crash (VLLM:: processes not killed). Launcher fix: added `pkill -9 -f 'VLLM::'` to cleanup |
| v11 | 1 | 2/5 | **32.6s/step confirmed** (step 1). XCCL intra-PG crash after step 1's weight sync — stale L0 from v9 (resolved in v12) |
| **v12** | 1 | **5/5** | **XCCL intra on fresh L0: 5/5 clean, exit=0.** Avg 56.1s/step. 2 syncs complete (27.8s at 2.2 GB/s each). Memory FLAT. Confirms v11 crash was stale L0. |
| **v13** | 1 | **5/5** | **Gloo intra fallback: 5/5 clean, exit=0.** Avg 72.1s/step. 2 syncs complete (64.8s and 65.8s at 0.9 GB/s). +28.5% vs XCCL intra. |

### RESOLVED: XCCL intra-PG broadcast crash (v11→v12)

The v11 XCCL intra-PG crash (`CCL_ERROR| comm_rank < comm_size`) was caused by
**stale L0 device state** from a prior v9 crash on the same nodes, NOT a fundamental
XCCL or oneCCL bug.

**Evidence:** v12 ran on a fresh job (clean L0 state) with identical code and config.
XCCL intra completed 5/5 steps with 2 successful intra-PG broadcasts, including
surviving the exact failure point (2nd sync round after FSDP training activity).

**Gloo intra-PG fallback** (`WSYNC_INTRA_METHOD=gloo`) was also validated (v13) as
insurance. It works but is 2.4x slower (0.9 GB/s vs 2.2 GB/s), adding ~40s to each
sync step's gen time. Non-sync steps are identical.

**v12 vs v13 comparison:**
| Step | XCCL Intra (v12) | Gloo Intra (v13) | Notes |
|------|------------------|------------------|-------|
| 0 | 79.8s | 78.1s | Cold start |
| 1 | 32.2s | 32.3s | No sync |
| 2 | 68.2s (gen=19.1s) | 110.6s (gen=60.9s) | Sync step |
| 3 | 29.9s | 29.9s | No sync |
| 4 | 70.4s (gen=20.7s) | 109.5s (gen=59.3s) | Sync step |
| **Avg** | **56.1s** | **72.1s** | +28.5% |

**Mitigation:** Ensure clean process cleanup between jobs (`pkill -9 -f 'VLLM::'`).
Use XCCL intra as production default.

### DP>1 per-replica PG fix — implemented, untested

The v9 deadlock root cause: with `VLLM_DP > 1`, one intra-PG covering ALL vLLM
workers (size=world_size-1) deadlocked because `/collective_rpc` dispatches
`receive_weights_xccl_streaming` per-replica. Each replica's TP workers entered
the broadcast independently, but the broadcast requires ALL `intra_size` workers.

**Fix (implemented 2026-04-28):** Per-replica PGs in both `vllm_weight_sync_worker.py`
and `grpo_full_finetune_distributed_xpu.py`:
- **Intra-PG**: `intra_size = tp_size_local` (was `world_size - 1`), prefix `wsync_intra_{replica_idx}`
- **Cross-PG**: One per replica, prefix `wsync_cross_{replica_idx}`, guard `tp_rank == 0` (was `my_rank == 1`)
- **Training**: Creates N cross-PGs, broadcasts to each sequentially

With N replicas, gloo broadcast is N× sequential. For 3 replicas × 60 GiB at
1.3 GB/s ≈ 138s. May need `vllm_weight_sync_interval=4+` or concurrent broadcast threads.

### 2-Chunk Rule (discovered 2026-04-29)

**G/fbs must be ≤ 2.** When the GRPO forward pass requires 3+ forward chunks
(G/forward_batch_size ≥ 3), CCL external memory explodes from ~1.8 GiB to
13-15 GiB at step 1. This is caused by FSDP AllGather/ReduceScatter buffer
retention across multiple model scans — 3 ref_fwd chunks trigger 3 full FSDP
model scans, and the intermediate CCL buffers from scans 1-2 are retained while
scan 3 runs.

With 2 chunks (G=32/fbs=16), external stays at ~2 GiB. With 1 chunk (G=16/fbs=16),
external is ~1.9 GiB. The jump to 3 chunks is catastrophic (not gradual).

**Production config envelope:**

| Config | Step time | Samples/s | Memory margin | Status |
|--------|-----------|-----------|---------------|--------|
| G=16, max_gen=128 | ~41s | 0.39 | 10+ GiB | Safe |
| **G=32, max_gen=128** | **~53s** | **0.60** | ~5 GiB | **Best throughput** |
| G=32, max_gen=192 | ~72s | 0.44 | ~1.5 GiB | Marginal |
| G=16, max_gen=512 | ~82s | 0.20 | ~0 GiB | Limit |
| G=48+, any max_gen | — | — | OOM/hang | Blocked (3+ chunks) |
| G=64, any | — | — | XPU bug | Blocked (kernel bug at batch=64) |

### Critical config

```bash
dp_replicate=1              # Pure 24-way FSDP (no replication)
forward_batch_size=16       # 1 AllGather round (CLI override, config says 4)
TORCHTUNE_USE_CHUNKED_LOSS=1
WSYNC_CROSS_METHOD=gloo     # Gloo TCP for cross-node PG
WSYNC_INTRA_METHOD=xccl     # XCCL for intra-node PG (default; gloo fallback at 2.4x cost)
vllm_weight_sync_interval=2 # Sync every 2 steps
CCL_PROCESS_LAUNCHER=none   # SSH + torch.distributed.run (not pmix)
GLOO_SOCKET_IFNAME=hsn0     # Slingshot NIC for gloo
```

### Launcher: `experiments/multinode_32b/run_32b_3node_24way.sh`

Key robustness features:
- `pkill -9 -f 'VLLM::'` cleanup (catches renamed worker processes)
- `curl --max-time 5` + `ssh -o ConnectTimeout=5` for health check (prevents infinite polling)
- DAOS model staging verification with checksum
- SSH pipe buffering: 24 ranks × verbose output through SSH pipes can stall log visibility (training continues at 190% CPU). Workaround: tail logs directly on training nodes.

### Memory (v11, step 1, 24 ranks)

- torch_alloc=15.72 GiB, torch_resv=42.29 GiB, gap=26.57 GiB
- POST-BWD: l0_free=19.80 GiB, external=1.90 GiB, retries=0, ooms=0

---

## Qwen3-30B-A3B Expert Parallelism Status (2026-04-30, v8 series)

First end-to-end EP=4/DP=3 attempts on the new Qwen3MoE infrastructure
(`Qwen3MoeTransformerLayer` + `qwen3_moe_ep_plan`). Run on 2-node dedicated-vLLM
topology: 12 train tiles + 12 vLLM tiles (3×TP=4). PBS jobs 8453156 (v1/v2) and
8453205 (v3-v7).

| Run | Knob change | Furthest op (train fwd) | Crash type |
|-----|-------------|-------------------------|------------|
| v1 | baseline | gloo timeout @ ~op #100 | timeout |
| v2 | gloo timeout 120s→1800s | op #427, layer ~13 | banned:1 PDE |
| v3 | Tier 1 env (CCL_ZE_CACHE=65536, unset XPU_USM_ALLOC_SO, gc:0.99) | op #425 | banned:1 PDE |
| v4 | + `torch.xpu.synchronize()` after H2D in EP collectives | op #427 | banned:1 PDE |
| v5 | `forward_batch_size=4` (disable chunking) | pass-2 OOM @ 60.83 GiB | clean Python OOM |
| v6 | `max_generated_tokens=128→64` (½ acts) | op #439, layer ~27 | banned:0 PDE |
| v7 | `max_gen=32 grpo_samples=4→2` (¼ acts vs v3) | op #253-equiv, layer ~30 | banned:1 PDE |
| v8a (3-node DP=6) | same as v7, +1 train node | op #261 | banned:1 PDE — same regime |
| v8g (3-node + MEMPROBE) | + per-EP-OP `mem_get_info`/`memory_allocated` probe | op #261 | **diagnostic: l0_free 46.7→0.17 GiB, torch_alloc 15.6→60.4 GiB, external FLAT at ~3.4 GiB → activation OOM, NOT IPC** |
| v8h (3-node) | + `forward_batch_size=1` | op #527 (full fwd + chunk-1 bwd done) → OOM at op #624 in chunk-2 fwd | activation OOM — chunk-1 grads pile on chunk-2 fwd |
| v8i (3-node) | + `forward_batch_size=1 grpo_samples=1` | **completed step 1, rc=0** | **first `loss=` line on Qwen3 EP** — loss=nan (max_gen=32 truncates response before reward signal; rewards=0.000 ⇒ zero-advantage divide), kl_loss=0.000984, ratios=1.000, resp_len=31 |
| v8j (3-node) | + `max_gen=64` | step 1, rc=0 | loss=nan (G=1 reward variance still 0); 44.1s/step (+33% vs mg=32) |
| v8k (3-node) | + `fsdp_cpu_offload=true G=2 fbs=1` | DNF (init >30 min) | FSDP2 cpu_offload H2D init too slow at 30B; left orphans → contaminated nodes |
| v8l/v8m (contaminated) | NSTEPS=5 / G=2 mg=16 | hung between gen and `chunk[0:1] fwd` | held-node contamination from v8k cleanup; mitigation = switch to fresh PBS job |
| v8m_b (fresh nodes) | `G=2 fbs=1 mg=16` | crash in `chunk[1:2] fwd`, banned:1 | **chunk[0:1] PRE-BWD = 50.08 GiB at mg=16 vs 50.85 GiB at mg=32 — halving max_gen reduced acts by <1 GiB**. Activations dominated by fixed buffers (KV cache, prompts, ref_logprobs), NOT response tokens. |
| v8n (fresh nodes) | `NSTEPS=2 G=1 fbs=1 mg=32` | step 0 clean; step 1 banned:1 in `chunk[0:1] fwd` | **PRE-STEP 1 alloc = 34.72 GiB** (vs PRE-STEP 0 = 14.06 GiB). `optimizer.zero_grad(set_to_none=True)` does NOT release FSDP2's internal unsharded grad buffers in EP mode → ~18.5 GiB grads carry into next step. |

### What we learned

- **Tier 1 env vars fix gloo CPU pressure** but not the GPU PDE crash. coll= time
  collapsed from 134s → 10s/layer in v3 (was monotonically growing 19s→134s in v2).
  `XPU_USM_ALLOC_SO` was leaking into v2 from the held shell — explicit `unset` is
  required in the launcher.
- **Async H2D use-after-free disproven** (v4): adding `torch.xpu.synchronize()` after
  the gloo→XPU H2D copy in `_ep_all_gather`/`_ep_reduce_scatter` did NOT fix banned:1.
- **No_grad → grad transition disproven** (v5): with chunking off, crash mode was
  clean Python OOM in pass 2, not banned:1. Ranks 3/7/11 (last in each EP group)
  hit OOM first → expert load imbalance is real but not the crash trigger.
- **Pure activation memory pressure disproven** (v7): a 4× activation reduction
  (max_gen 128→32 + grpo_samples 4→2) gained only ~3 layers vs v6's 2× reduction
  (layer 27→30). Diminishing returns rule out pure memory pressure.
- **`_AllGatherRS`/`_ReduceScatterAG` save no tensors for backward** — only
  `ctx.group`. So saved-tensor pinning of bounce-CPU buffers is also out.
- All crash addresses are in the L0 IPC range (`0xff01...`) and always occur
  inside the **train fwd** (the fwd with autograd save). Ref/policy fwds (no_grad,
  same EP code path) complete cleanly every run.

### Working hypothesis (REFUTED 2026-04-30 by v8g MEMPROBE)

Earlier hypothesis was L0 IPC handle pressure during autograd graph build over EP
collectives. **v8g per-EP-OP memory probe disproved this**: `external = l0_used -
torch_alloc` stayed FLAT at ~3.4 GiB through all 261 ops, while `torch_alloc` walked
linearly from 15.6 → 60.4 GiB driven by saved-activation accumulation. The "banned:1
PDE" signature is just how XPU surfaces an L0 OOM during a CCS write that overruns
the allocation map.

### Resolution (2026-04-30)

v8h cut `forward_batch_size 2→1` (halves saved activations per fwd pass): completed
all 48 layers fwd + chunk-1 bwd, then OOMed in chunk-2 fwd (chunk-1 grads stayed live).
v8i added `grpo_samples 2→1` (single chunk): **first `loss=` line on Qwen3 EP**, rc=0.

Loss is `nan` because `max_gen=32` truncates the response before any GO term reward
match — rewards=0.000 → zero-advantage divide. Not an infrastructure bug; it's the
activation-budget tradeoff.

**Production envelope for Qwen3 EP=4/DP=6** (current code, `EP=4/DP_replicate=6/DP_shard=4`):
`forward_batch_size=1 / grpo_samples=1 / num_steps=1 / max_generated_tokens ∈ {32, 64}`
with ~8 GiB headroom at the 48-layer fwd peak. **Both G≥2 and NSTEPS≥2 are blocked.**

### Unified architectural blocker (v8m_b + v8n share root cause)

v59's `reduce_grads=False` (`grpo_full_finetune_distributed_xpu.py:1440-1464`)
forces FSDP2 to skip `post_backward` reduce_scatter on every FSDPParamGroup in
EP mode. The full unsharded grad pool (~18.5 GiB at DP_shard=4) stays resident:
- between chunks within a step → blocks G≥2 (v8m_b chunk-2 fwd OOM)
- between steps → blocks NSTEPS≥2 (v8n step-1 chunk-0 fwd OOM, with PRE-STEP 1
  allocation already at 34.72 GiB before any new activations are saved)

`optimizer.zero_grad(set_to_none=True)` at lines 3047/3322 does not reach
FSDP2's internal grad buffers — those grads live in `FSDPParamGroup` state, not
in `optimizer.param_groups[*]['params']`. Halving `max_gen` (v8m_b 32→16) only
reduced per-chunk activations by <1 GiB because activations are dominated by
fixed-cost buffers (KV cache for prompt+ref, ref_logprobs storage, prompt
sequence buffers).

The current path's grad sync is a single deferred XCCL all_reduce on the
dp_replicate group AFTER all chunks (lines 2945-2954). Unblocking BOTH G≥2 AND
NSTEPS≥2 requires explicit `fully_shard`-aware reduce_scatter immediately after
each chunk's bwd AND after the optimizer step, sequenced with the existing
deferred all_reduce so grads aren't double-reduced. **Code change required —
not a tuning fix.**

### Diagnostic rule (kept for future XPU EP work)

For any future XPU `banned:0/1 PDE` crash with addresses in `0xff0[0-5]...`:
1. Add `_ep_mem_probe`-style trace (`torchtune/modules/moe/_parallelism.py:108-135`)
   to localize whether `torch_alloc` or `external` is climbing.
2. If `external` is FLAT, it's an activation OOM and the fix is per-tile budget
   (`forward_batch_size`, `grpo_samples`, `max_generated_tokens`, AC), not env
   vars or PG transport changes.
3. The PDE signature is a red herring; XPU surfaces L0 OOM as a CCS-write fault
   that overruns the allocation map.

### Comparison with Gemma4 EP (v141–v161)

Different failure mode. Gemma4's blockers were:
- **Backward desync** (v153) — autograd hook ordering not deterministic per rank
- **ScatterAdd shape mismatch** (v154) — router non-determinism under non-reentrant AC
- **AsymmetricAG-BWD deadlock** (v158) — fixed by autograd anchors in v161

Qwen3 EP has none of these (forward and AG-FWD/RS-FWD all succeed in lockstep).
The Qwen3MoE codepath (`Qwen3MoeTransformerLayer.use_reentrant=False` with MoE
outside the AC region, `bmm_scatter` expert forward) sidesteps the Gemma4 saga
entirely. The remaining blocker is purely **train fwd activation/IPC pressure**,
not autograd ordering.

### Production status

EP for Qwen3-30B-A3B reached **first `loss=` line** at G=1/fbs=1/NSTEPS=1
(v8i, 2026-04-30) but is **not yet a viable training config** — both G≥2
(needed for advantage variance) and NSTEPS≥2 (needed to actually train) are
blocked by the unified architectural issue above. Production RL training
continues on EP=1 (replicated experts) — the validated G=8 single-node config
(54.8s/step, 9.2 tok/s, 2026-04-27).

Full writeup: `docs/reports/qwen3_ep_v8_3node_20260430.md`. Memories:
`project_qwen3_ep_v1_v2.md` (v1-v2), `project_qwen3_ep_v3_v7.md` (v3-v7),
`project_qwen3_ep_v8_3node.md` (v8 series). Launchers:
`experiments/ep_parallelism/hold_qwen3_ep_v{1-7,8}.sh`. Underlying recipes:
`recipes/dev/run_qwen3_30b_ep4_vllm_2node.sh` (v1-v7),
`recipes/dev/run_qwen3_30b_ep4_vllm_3node.sh` (v8 series).

---

## Qwen3-30B-A3B EP=8 Status (2026-04-30, v10 series)

PBS jobs 8460378 / 8461394 (capacity, 2 nodes). Single training node, EP=8
/ dp_replicate=1 / dp_shard=8 (8 of 12 tiles), 3rd-node vLLM 3×TP=4. The
v9 `_ep_release_fsdp_unsharded_grads` helper plus
`torchtune.dev.bioreason.optim.AdamWBf16` together unblock the v8 unified
blocker.

| Run    | Config                                    | PRE-STEP-0 | PRE-STEP-1 | Outcome |
|--------|-------------------------------------------|------------|------------|---------|
| v10b2  | G=2 NSTEPS=1 FBS=1                        | 7.19       | n/a        | **PASS**; loss=0.0061; chunked accumulate validated |
| v10c   | G=4 NSTEPS=3 FBS=1, plain AdamW           | 7.19       | 29.99      | step-2 fwd OOM — fp32 momentum+variance pinned floor |
| v10d   | G=1 NSTEPS=3 FBS=1, plain AdamW           | 7.19       | 29.96      | step-2 fwd OOM same as v10c — proves AdamW is the wall |
| v10e   | G=2 NSTEPS=2 FBS=1, **AdamWBf16**         | 7.19       | **15.77**  | **PASS rc=0**; both steps finite |
| v10f   | G=4 NSTEPS=3 FBS=1, **AdamWBf16**         | 7.19       | 15.77 / 15.47 | **PASS rc=0**; ~3:20/step (192-204s); per-step sweep confirmed |

### Per-step wall clock (v10f average over 3 steps)

| Phase | Time | Share |
|-------|------|-------|
| gen (vLLM HTTP)             | ~5.6 s    | 3%   |
| grpo fwd                    | ~3.8 s    | 2%   |
| grpo bwd                    | ~5.6 s    | 3%   |
| **v9 release helper (gloo)** | **~99 s** | 49%  |
| clip                        | ~0.1 s    | <1%  |
| **optimizer (AdamWBf16 CPU)** | **~85.8 s** | 43%  |

~93% of step time is CPU-side workarounds: gloo bounce in the helper and
CPU AdamWBf16. Compared to the validated EP=1 single-node path
(54.8s/step at G=8, 2026-04-27), EP=8 is **3.6× slower per step at half
the rollout count**.

### Production envelope

Single-node EP=8 dp_replicate=1, current code:
- Optimizer **must be** `torchtune.dev.bioreason.optim.AdamWBf16`.
- `forward_batch_size=1` is required to exercise the chunked-accumulate
  path under `G≥2` — `fbs=2` collapses `G=2` into a single chunk and
  hits activation OOM (v10b).
- Validated to `G=4 NSTEPS=3` with finite losses and PRE-STEP-N flat.
  Reward curve not yet run.

### Open items (not yet committed)

A. **3-node EP=8 dp_replicate=3 / dp_shard=8** — would test whether v75
   XCCL cross-replica sync works with the real grads the helper now
   produces, and whether 3-way replication restores enough headroom to
   put plain AdamW back on device.
B. **Rewrite the v9 helper to use XCCL reduce-scatter on-device** —
   would directly remove the dominant ~99s/step cost. Requires
   re-validating that the global barrier alone is sufficient ordering
   for the v59 race that motivated CPU gloo bounce in the first place.
C. **G≥8 sweep at NSTEPS=1** — PRE-STEP-1 had ~7 GiB headroom on v10f at
   G=4; G=8 may still fit before any further intervention.

Full writeup: `docs/reports/qwen3_ep_v10_20260430.md`. Memories:
`project_qwen3_ep_v9_helper.md`, `project_qwen3_ep_v10_unblocked.md`.
Launchers: `experiments/ep_parallelism/hold_qwen3_ep8_v10{a,b,b2,c,d,e,f}.sh`.
Underlying recipe: `recipes/dev/run_qwen3_30b_ep8_vllm_2node.sh`.

---

## Prioritized Next Steps

### Tier 1: Immediate (production training)

**1. ~~Qwen3B gene recall production run~~ (DONE 2026-04-24)**
- Job 8449766: 130 steps clean with `usm_caching_alloc.so` (7h walltime, PBS SIGTERM)
- Reward: noisy, peak 43.75% success (steps 44, 84). No monotonic convergence over 130 steps.
- KL well-controlled: approx_policy_kl = 0.0003-0.0007 range
- 3 checkpoints saved (epoch_0/1/2 in `outputs/gene_recall_production/ref/`)
- **Next**: longer run (500+ steps), or curriculum/G adjustments if plateau persists

**2. ~~32B dedicated-vLLM 2-node + 2-hop XCCL~~ (VALIDATED 2026-04-24)**
- Job 8450367: **24/24 steps clean** (gc:0.6, walltime-limited). Genuine steady state.
- Memory: 62.04 GiB FLAT from step 7 (17 consecutive steps). OFI MR=1.56 GiB stable.
- Script: `experiments/multinode_32b/run_32b_2hop_production.sh`
- **Next**: longer production run (50-100+ steps) to confirm indefinite stability

**3. ~~3-node 24-way FSDP — XCCL intra-PG crash~~ (RESOLVED 2026-04-28)**
- v12 (XCCL intra, fresh L0): **5/5 clean, 56.1s/step avg**. v11 crash was stale L0, not XCCL bug.
- v13 (gloo intra fallback): **5/5 clean, 72.1s/step avg**. XCCL intra 2.4x faster (2.2 vs 0.9 GB/s).
- **Next**: Validate DP>1 per-replica PGs (VLLM_DP=3); long production run (50+ steps)
- Script: `experiments/multinode_32b/run_32b_3node_24way.sh`

**4. Qwen3-30B-A3B EP=4 — unblock G≥2 / NSTEPS≥2 (CODE CHANGE) (NEW 2026-04-30)**
- v8 series reached first `loss=` line (G=1/fbs=1/NSTEPS=1) but is not yet trainable: G≥2 needed for advantage variance, NSTEPS≥2 needed to actually train.
- Both blocked by v59's `reduce_grads=False` keeping ~18.5 GiB FSDP2 unsharded grads alive across chunk and step boundaries.
- Concrete patch: in `recipes/dev/grpo_full_finetune_distributed_xpu.py`, add explicit `fully_shard`-aware reduce_scatter immediately after each chunk's bwd (~line 2945-2954) AND after the optimizer step (~line 3047, 3322), sequenced with the existing deferred dp_replicate all_reduce so grads aren't double-reduced.
- Validation plan once landed: rerun v8m_b config (G=2 mg=16) and v8n config (NSTEPS=2) on a fresh 3-node hold; need PRE-STEP N alloc to drop back near params-only (~14 GiB) and chunk[1:2] PRE-FWD alloc to drop by ~18.5 GiB.
- Full diagnosis: `docs/reports/qwen3_ep_v8_3node_20260430.md`. Memory: `project_qwen3_ep_v8_3node.md`.

**5. Gemma4 31B gene recall — fix training stability first**
- Root cause 1: no SFT warm-up for `<genes>` format → add 5–10 SFT examples as prompt prefix, or cold-start with supervised fine-tuning on 20 examples
- Root cause 2: max_generated_tokens=512 causes all responses to be truncated → reduce to 256, or better: fix EOS generation by adjusting stop tokens config
- Root cause 3: lr too high for warm-up — step 2 grad_norm=3968 suggests initial lr is too aggressive; try linear warm-up over 20 steps
- Only attempt RL training after steps 1-3 stabilize KL

### Tier 2: Infrastructure improvements

**6. 3B gene recall: improve convergence**
- Current: 130 steps, noisy reward, peak 43.75% success, no monotonic improvement
- Options: increase G (16→32), curriculum (easy genes first), SFT warm-up, longer runs (500+)
- Checkpoints available in `outputs/gene_recall_production/ref/` for resume

**7. 50-100 step 32B validation**
- 24 steps shows steady state; a longer run would confirm indefinite stability
- Use `experiments/multinode_32b/run_32b_2hop_production.sh` with NSTEPS=100, longer walltime

**8. ~~Single large memmove for 31B~~ (SUPERSEDED)**
- 2-hop XCCL replaced SHM for 32B weight sync. SHM memmove optimization no longer on critical path.
- SHM still used for 3B but sync is already hidden (1.4s waited). Low priority.

**9. ~~Test XCCL broadcast as SHM replacement~~ (DONE)**
- 2-hop XCCL is now the production weight sync method for 32B (9.1s sync, 7.9 GB/s).
- Implemented in `vllm_weight_sync_worker.py` + recipe. SHM eliminated for 32B.

### Tier 3: Research / deferred

**10. 1-step gather overlap** (reduce 31B waited from 13s to ~7s)
- Restructure training loop to start gather immediately after optimizer step
- Feed gathered weights to vLLM during GRPO forward of the SAME step
- 1-step weight lag is standard in RL — no algorithmic concern

**11. EP backward AllToAll resolution** (deferred)
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

---

## 2026-04-24: Production Runs + Late-step banned:1 Root Cause

### 2-hop XCCL weight sync — VALIDATED and PRODUCTION-READY

Test CK (2026-04-24): 3/3 steps clean (initial validation). Extended to **24/24 steps clean**
(job 8450367, gc:0.6 + 2-hop XCCL, terminated by PBS walltime). Sync: 38s → 9.1s (4× speedup).
Step time: 42.5s avg (range 39.9-47.1s). Memory: genuine steady state (62.04 GiB FLAT from step 7,
OFI MR=1.56 GiB stable). See `experiments/multinode_32b/test_ck_2hop_wsync.sh`,
`experiments/multinode_32b/run_32b_2hop_production.sh`.

Production launcher: `experiments/multinode_32b/run_32b_2hop_production.sh`
- Architecture: Node 0 = 3× vLLM replicas (TP=4, 12 tiles), Node 1 = FSDP2 training (12 tiles)
- 2-hop protocol: training → vLLM rank 1 via 2-rank Slingshot broadcast; rank 1 → ranks 2–12 via XeLink
- Measured: ~9.4s sync, ~43–48s/step

### Late-step banned:1 root cause (2026-04-24)

**Symptom** (without mitigation): Reproducible GPU page fault at a specific late step:
- 32B: crashes at step 7-10 (without gc:0.6). **Fixed**: gc:0.6 runs 24+ steps clean.
- Gene3b: crashes at step 7 (without caching alloc). **Fixed**: usm_caching_alloc.so runs 130+ steps clean.

**Mechanism** (PyTorch internal allocator GC):
1. PyTorch's caching allocator accumulates reserved memory each step (gene3b: 42→50→58 GiB steps 4–6; 32B: plateaus at 62 GiB from step 7 onward)
2. At the critical step, a new allocation can't be satisfied from the cached free blocks (fragmentation or cache full); the allocator triggers internal GC
3. GC calls `sycl::free` on L0 allocations in its cache, returning them to the driver
4. Some freed L0 blocks are FSDP2 AllGather SEND buffers that oneCCL has open IPC handles for
5. Next FSDP2 AllGather: CCL accesses the stale IPC handle → GPU page fault → `banned:1`

**NOT caused by explicit `device_empty_cache()` calls** — those are already monkey-patched to a no-op on XPU at line 722 of the recipe (`pass` branch for `device.type == "xpu"`).

**Mitigation (3B): `usm_caching_alloc.so` pluggable allocator**

Blocks ≤ 8 GiB (`kBucketCap`) are permanently pooled (never freed to L0 driver). This covers FSDP2 per-layer shards (~81-94 MB/tile), XCCL weight sync buffers (~6 GiB, rounds to 8 GiB bucket), and all standard FSDP AllGather outputs. Allocations > 8 GiB are freed to L0 immediately after use.

Applied via `XPU_USM_ALLOC_SO` env var:
```bash
XPU_USM_ALLOC_SO="${TT_DIR}/recipes/dev/usm_caching_alloc.so"
```

Launchers:
- `experiments/gene_recall/run_3b_gene_recall_production.sh` — CONFIRMED (job 8449766, 130 steps clean, 7h PBS walltime)

**FAILS at 32B** — all pluggable allocator variants crash at step 0-1:
- Gen1 (`usm_caching_alloc.so`): Power-of-2 bucketing → OOM at step 4 (2× waste on 21 MiB optimizer tensors)
- Gen2 (`usm_arena_alloc.so`): Missing `queue->wait()` → step-1 cross-stream crash
- Gen3 (`usm_caching_alloc_v2.so`, 2026-04-25): Fixed cross-stream + exact alignment, but per-tensor L0 allocs exhaust 64 GiB HBM. OOM retry releases cached blocks → VAs invalidated → CCL `banned:1` at step 1 (same failure mode as PyTorch GC, just faster)

**Root cause (pluggable allocator failure at 32B)**: Per-tensor `sycl::malloc_device` creates individual L0 allocations (~1800 large + ~3500 small per tile). The default allocator's `expandable_segments` suballocates from large segments (fewer L0 VAs), so GC can reclaim suballocations without invalidating the parent segment's VA. Pluggable allocators bypass this mechanism entirely.

**Mitigation (32B): Default allocator with `gc:0.95` — VALIDATED 5/5 STEPS CLEAN (2026-04-25)**

`PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95`

**Validated 5/5 steps clean** (job 8450884, 2-node, 2-hop XCCL weight sync, G=16, max_gen=128):
- Steps 0-4: zero crashes, zero retries, zero OOMs
- Timing: 40.5s/step average (step 0: 40.1s, steps 1-4: 37.8-41.7s)
- Memory (rank 0):
  - Step 0 PRE-BWD: l0_free=28.52, torch_resv=33.87, external=1.60 GiB
  - Step 2-4 PRE-BWD: l0_free=2.53-2.58, torch_resv=59.64, external=1.76-1.81 GiB (FLAT)
- GC never fired (pool satisfies all requests from cache after step 1 warmup)
- Between-step memory: allocated=28.45 GiB, reserved=59.13 GiB — matches production baseline

Also validated with gc:0.6 (job 8450367, 24 steps clean, same profile). GC threshold is irrelevant when pool is warm — any value works as long as the pool satisfies all requests from cache.

**Required CCL env var** (both mitigations): `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536`
- Prevents early eviction at step 2 (default 1000 fills after ~28 steps of AllGather registrations)

### FI_MR_CACHE_MONITOR=userfaultfd — DOES NOT PREVENT BANNED:1 (2026-04-25)

**Single-node (XeLink only):** 5/5 steps clean (job 8450884). Timing 40.1s/step — zero
regression. Memory identical to `disabled`.

**Cross-node (Slingshot):** 7/8 steps then `banned:1` at step 8 (job 8450921). userfaultfd
is compatible with CXI/Slingshot but does NOT prevent the crash:

- userfaultfd manages OFI MR cache (RDMA memory registrations, user-space, libfabric)
- `banned:1` comes from CCL IPC handle cache (L0 driver-level, kernel-space, XeLink peer access)
- These are completely separate mechanisms — userfaultfd cannot invalidate L0 IPC handles
- Worse: userfaultfd caused rank 0's torch_resv to expand to 62.54 GiB (vs 59.64 with disabled),
  triggering GC at step 7 → banned:1 at step 8. With `disabled`, GC never fired in 5 steps.

**Production setting: `FI_MR_CACHE_MONITOR=disabled`** — provides the best outcome
(no GC trigger, stable 59.64 GiB torch_resv).

### Long-run 32B stability: ~80 step limit from external CCL growth

Memory is flat from step 2 onward (torch_resv=59.64 GiB). External CCL memory grows at
~30 MiB/step (1.60→1.81 GiB over 5 steps). At this rate, l0_free (2.5 GiB) depletes at
~step 85. Beyond that, CCL's behavior at l0_free=0 is uncertain — if it needs fresh L0
allocations during a collective, the crash returns as UR:40.

**Mitigation:** `save_every_n_steps=20` with checkpoint-restart at ~step 60-70.

**Root cause identified (2026-04-25):** FSDP2 external growth diagnostic (job 8450943)
ran 100 steps of pure FSDP2 fwd/bwd (no GRPO, no weight sync, no optimizer) on Qwen3-32B.
External memory was **dead flat at 1.68 GiB** — zero growth across all 100 steps. This
proves the 30 MiB/step growth is NOT from FSDP2 AllGather/ReduceScatter but from
GRPO-specific operations. Primary suspect: XCCL 2-hop weight sync broadcast (61 GiB/step).
Other candidates: clip_grad_norm AllReduce, XCCL process group management.

This is actionable: increasing `weight_sync_interval` (e.g., sync every 2 steps instead of
every step) would halve the external growth rate and extend safe run length to ~160 steps.
The async deferred weight sync (validated at interval=2, 13.4% throughput improvement)
already provides this benefit.

### Side note on MEMPROBE with caching alloc active

When `XPU_USM_ALLOC_SO` is set, `torch.xpu.mem_get_info()` raises `NotImplementedError` and `torch.xpu.memory_stats()` returns {} (monkeypatched). MEMPROBE shows `l0_free=-1.0, torch_alloc=0.0`. Memory monitoring is blind with pluggable allocator active.
