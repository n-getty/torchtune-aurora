# Per-Tile Colocate (vLLM + FSDP Trainer on the Same XPU Tiles)

**Status as of 2026-05-08 (Qwen3-8B reference workload)**

| Path | Architecture | Steady-state | Production-ready | When to use |
|------|--------------|--------------|------------------|-------------|
| **W2 — process-isolated 8+4** | Trainer FSDP2 ZeRO-3 on tiles 0..7, vLLM TP=4 on 8..11 (no per-tile co-tenancy) | **247 s/step** | **YES** — 10/10 clean, exit=0 | Default. Pick this whenever 4 spare tiles exist. |
| **W17+W19 — kill+respawn fully-colocated** | Trainer FSDP2 on tiles 0..7, vLLM TP=8 Ray actors on the same 0..7 tiles. vLLM is killed before each BWD and respawned before each wsync. | **~290 s/step** (3/3 steps clean) | YES (correctness) / NO (wsync architecture) | Only when HBM forces full 8-tile colocation. wsync cost is structurally bad on this path — see §5. |
| **`colocate_sleep` same-process** | Trainer FSDP2 + vLLM in the same Python process on all 12 tiles (TP=1 per rank); vLLM `sleep()`/`wake_up()` between gen and train | **~27 s/step (8B)** | **YES (3-step limit)** — 4/4 clean, crash at step 4-5 from L0 UR leak | Best single-node path. 9× faster than W2. Use checkpoint/restart every 3 steps for longer training. |

This document synthesizes the W4-W19 finding chain, the L0 co-tenancy mechanism, the W19 wsync-architecture limitation, and the open path forward (L0 IPC handles or upstream fix). It supersedes:

- `docs/reports/colocate_tp8_status_20260506.md` (Phase 1/2 external_launcher elimination — frozen pre-Ray)
- `docs/reports/colocate_ray_tp8_status_20260507.md` (Phase 4 Ray status — frozen at W2 PASS, pre-W17)
- `docs/bugs/xpu_l0_event_pool_co_tenancy.md` is still the authoritative deep-dive on the UR40 mechanism; this document is the architectural / decision-making layer above it.

---

## 1. Why per-tile colocate matters

The default RL serving topology on Aurora is **process-isolated**: trainer FSDP on N tiles, vLLM on a separate set of M tiles, weights shipped between them via gloo or XCCL broadcast. This works (W2 above), but it costs M tiles you cannot use for training and forces every weight sync onto the inter-process fabric. For a 12-tile node with an 8-tile trainer, M = 4 means you're spending 33% of the node on inference.

**True per-tile colocate** is the alternative: trainer + vLLM share the same tile, swap roles in time (gen → train → gen → ...), and ideally share weights via on-device pointer/IPC rather than shipping bytes across processes. This is what NeMo-RL and TorchRL implement on CUDA via vLLM's `sleep`/`wake_up` API in a single process — wsync collapses to a `tensor.copy_()` against a resident buffer (~1-3 s for 16 GB at PVC HBM bandwidth).

Aurora **can** deliver this profile via `colocate_sleep` mode — validated 2026-05-09 on Qwen3-8B at ~31 s/step (8× faster than process-isolated W2). The critical fix was `num_gpu_blocks_override` to prevent bimodal KV cache sizing from the L0 `mem_get_info` bug. Details in §9 below.

The multi-process colocate path (Ray TP=8) cannot deliver this profile. The reasons are documented below.

---

## 2. The L0 co-tenancy mechanism

When two distinct OS processes attach to the same Aurora PVC tile and both submit Level-Zero work concurrently, the per-tile L0 driver context exhausts an internal event/queue/submission resource pool **at the very first backward kernel of step 0**, with **~30 GiB of HBM headroom remaining**.

- Error: `level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)`.
- Same UR error code as the well-known `empty_cache()` + FSDP `storage.resize_()` leak, but a completely different trigger and timing.
- No documented env knob for the pool ceiling. The leak surfaces only with two L0 clients on the same tile.

**Mechanism breakthrough (W17, 2026-05-07)**: the wedge is held by **the act of two L0 clients concurrently submitting work to the same tile**, not by per-process driver-init bookkeeping. Tearing down vLLM + `ray.shutdown()` after generation but before BWD unblocks BWD on the same trainer process and the same per-tile L0 driver state that wedged in every prior probe.

**Symmetric corollary (W18 hang, W19 fix)**: the wedge applies in both directions. While vLLM TP=8 is initializing (posting `all_reduce` for tensor-parallel coordination), trainer ranks must hold zero L0 work on the same tiles. A naive `torch.distributed.barrier()` (XCCL) on the trainer side deadlocks vLLM's init. The fix is a CPU-only gloo barrier (`self._ray_colocate_gen_pg`) for the trainer wait.

Authoritative bug doc: `docs/bugs/xpu_l0_event_pool_co_tenancy.md`. Upstream filing draft: `docs/bugs/UPSTREAM_FILING_DRAFT_l0_resource_pool.md` (frozen pre-W17; needs W17/W19 evidence appended before filing).

---

## 3. The two phases of attempted colocate

### Phase 1/2 — `external_launcher` colocate (ELIMINATED)

vLLM's `external_launcher` distributed-executor backend launched vLLM workers as additional ranks of the trainer's torchrun. Wedged deterministically at the **first** vLLM `_gather_logits` allgather inside step-0 generate — py-spy traced to `ze_handle_exchange_entry::common_fd_mode_exchange` (Level Zero IPC handle FD socket exchange).

E32 (keep-PG-alive), E33 (sockets), and E34 (raw 128 MiB pre-warm) all failed to clear it. This path was eliminated in favor of Phase 4. See `docs/reports/colocate_tp8_status_20260506.md`.

### Phase 4 — Ray-colocate (CURRENT)

vLLM uses `distributed_executor_backend="ray"`. Trainer rank 0 spawns 8 vLLM Ray actors (one per tile) via `LLM(...)`. Each Ray actor is a separate OS process and uses Ray's intra-node SHM object store for IPC. This sidesteps the L0 IPC FD-socket allgather wedge from Phase 1 — generation completes cleanly. **But it introduces the per-tile co-tenancy wedge described in §2** because trainer and Ray actor share each of the 8 tiles. Hence the W4-W19 probe sequence below.

---

## 4. W-probe finding chain (W4 → W19)

The full mitigation table is in `docs/bugs/xpu_l0_event_pool_co_tenancy.md`. Summary of which axis each probe attacked and what it ruled out:

| Probe | Attack axis | Result |
|-------|-------------|--------|
| W2 | **Avoid co-tenancy entirely** (process isolation 8+4) | **PASS** — production fix |
| W4 | empty_cache between gen and BWD | FAIL (L0 events are not what's exhausted) |
| W5 | per-rank L0 driver contexts | FAIL |
| W6 | counter events | FAIL |
| W8 | IPC backing change | FAIL — rules out IPC handle table |
| W9 | event-pool count | FAIL |
| W10/W11 | cmdlist mode probing | FAIL (W11 shifts UR40 → PDE; mode-specific) |
| W11a | per-thread immediate cmdlists | FAIL (mode=2 emits same UR40) |
| W12 | `ZE_FLAT_DEVICE_HIERARCHY=COMBINED` | FAIL (true HBM OOM under different topology — driver-topology-dependent) |
| W13 | oneCCL persistent SYCL temp buffers | FAIL — wedge is BELOW oneCCL |
| W15 | CCS=4 on all 12 tiles (W5 corrected) | FAIL — rules out CCS partition mismatch |
| W16 | cmdlist-pool ceilings (events/cleanup/batch) | FAIL — wedge is in EXISTENCE of the resource family, not its capacity |
| SKIPGEN | actor exists but never calls generate | FAIL — confirms idle co-tenancy is enough |
| **W17** | **kill vLLM after gen, before BWD** | **PASS BWD+opt** — sequential co-tenancy unblocks the wedge |
| W18 | W17 + respawn vLLM with xccl barrier | HUNG at vLLM `init_device → all_reduce` (symmetric wedge) |
| **W19** | W18 + gloo barrier on `_ray_colocate_gen_pg` | **PASS 3/3 steps** — kill+respawn cycle works |

The fundamental conclusion: **W4-W16 attacked the wrong axis**. None of them changed the live concurrent co-tenancy state, so none moved the symptom. W17 was the first probe to change that state, and it cleared the wedge immediately.

---

## 5. The W19 wsync architecture failure

W17+W19 makes fully-colocated TP=8 **work end-to-end** (correctness PASS, 3/3 steps clean, ratios=1.0000). It does **not** deliver the colocate performance benefit. This section explains why.

### 5.1 Per-step timing breakdown (Qwen3-8B G=4 NSTEPS=3)

Average over the three steady-state steps:

| Phase | Time | Share |
|-------|------|-------|
| **gen** (vLLM TP=8, 256 seqs, ~110k tok @ ~830 tok/s) | **149 s** | 51 % |
| **wsync_gather** (umbrella, contains the W19 cycle below) | **125 s** | 43 % |
| └─ W17 teardown (`ray.shutdown()` + `del self._vllm_llm`) | 1.4 s | 0.5 % |
| └─ drain (`TORCHTUNE_RAY_COLOCATE_DRAIN_S`, default 5) | 5 s | 1.7 % |
| └─ W19 respawn (vLLM TP=8 cold-start, warm cache) | 30.5 s | 10.5 % |
| └─ wsync RPC (399 FSDP `full_tensor` + serial `collective_rpc` to 8 actors) | **88-97 s** | 31 % |
| **grpo_step** (BWD + optimizer) | 16 s | 6 % |
| clip / opt / lr_sched | 0.1 s | 0 % |
| **TOTAL** | **~290 s** | |

### 5.2 Where the cost actually lives

**The W19 cycle itself adds only ~36 s** (kill 1.4 + drain 5 + respawn 30). That's +14 % on top of the W2 baseline. The remaining ~7 s of the 43 s gap is within wsync.

**The dominant cost is the wsync RPC, not the kill+respawn cycle.** 92 s for 16 GB across 399 params ≈ **170 MB/s**. The path is:

1. All 8 trainer ranks call FSDP `full_tensor()` per param → XCCL all-gather across 8 tiles.
2. Rank 0 calls `self._vllm_llm.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, data))` to fan out to the 8 vLLM Ray actors.
3. Each actor receives via Ray intra-node SHM object store, deserializes, and copies into its TP shard.
4. **399 round-trips, serial.**

This is **server-mode wsync done through Ray RPC**. The trainer and the vLLM actor are physically on the same tile, but the data still gets marshaled, IPC'd, and copied across process boundaries. Per-param Ray RPC roundtrip overhead dominates.

### 5.3 Why proper colocate wsync is structurally impossible on this path

The "fast" colocate paths used by NeMo-RL, TorchRL, and `colocate_sleep` all rely on trainer + vLLM in the **same process**, so weights are pointer-shared / SHM-copied. Wsync becomes `param.data.copy_(trained_weight)` against a resident buffer — DDR/HBM bandwidth, no marshaling, no RPC. Expected ~1-3 s for 16 GB.

W19 cannot reach that profile because:

- **Ray-colocate is N+1 separate processes** (trainer + 8 vLLM TP ranks). The only on-device shortcut would be **L0 IPC handles** between the trainer process and each Ray actor process on the same tile. Not implemented; would need new plumbing.
- **W17 kill loses all vLLM state every step.** Even if L0 IPC handles existed, they would be invalidated on every respawn. Persistent IPC handles + W17 kill are mutually exclusive.

### 5.4 Options for getting wsync below 90 s

| Option | Estimated wsync | Engineering | Notes |
|--------|------------------|-------------|-------|
| (a) Batch RPC — pack all 399 params into one `collective_rpc` with a single buffer (analogous to `TORCHTUNE_EP_WSYNC_LAYER_BATCH`) | 10-30 s | Small | Cuts 399 round-trips → 1. Best near-term lever. Still nowhere near 3 s. |
| (b) L0 IPC handles between trainer and Ray actor processes | 1-3 s | Significant | Requires per-param `ze_command_list_append_memory_copy` plumbing. Incompatible with W17 kill cycle (handles invalidated). Only viable if the underlying L0 wedge is fixed and W17 is no longer needed. |
| (c) Fix `colocate_sleep` upstream — eliminate the L0 wedge so same-process colocate works | 1-3 s | Out of scope (Intel L0 driver) | The right long-term answer. Bug filed; see `UPSTREAM_FILING_DRAFT_l0_resource_pool.md`. |
| (d) Stay on W2 (8+4 process isolation) | 247 s/step total, ~10-30 s wsync via XCCL broadcast | None | Already in production. ~17 % faster than W19 today. |

Recommendation: implement (a) opportunistically as a wsync-method flag; do not invest in (b) until upstream L0 is fixed; treat W19 as a **correctness path** for HBM-constrained workloads, not a performance path.

---

## 6. Operating the W17+W19 path

### 6.1 Knobs

| Env var | Default | What it does |
|---------|---------|--------------|
| `TORCHTUNE_RAY_COLOCATE_KILL_AFTER_GEN` | `0` | Master gate. Set `1` to enable W17 kill + W19 respawn cycle. |
| `TORCHTUNE_RAY_COLOCATE_DRAIN_S` | `5` | Seconds all ranks sleep after `ray.shutdown()` so i915 + L0 reclaim per-client per-tile state. Lower bound is empirical; 2 s may work, 0 s does not. |

### 6.2 Recipe hooks

- W17 (kill before BWD): `recipes/dev/grpo_full_finetune_distributed_xpu.py:2807` — gated on `TORCHTUNE_RAY_COLOCATE_KILL_AFTER_GEN=1`. After gen and qr broadcast, rank 0 calls `del self._vllm_llm` + `ray.shutdown()`; all ranks `time.sleep(DRAIN_S)`; xccl barrier (safe — vLLM is dead, no concurrent L0); enter BWD.
- W19 (respawn before wsync): `recipes/dev/grpo_full_finetune_distributed_xpu.py:4150` — same gate. Rank 0 calls `_init_vllm_ray_colocate(self._cfg)` (idempotent: sets `self._vllm_llm = None`, `ray.init(address=...)` if needed, builds new `LLM(...)`). All ranks then barrier on `self._ray_colocate_gen_pg` (gloo, CPU-only). Then `_sync_ray_colocate_weights` runs against fresh actors.

### 6.3 Hard rules

- The post-respawn barrier MUST be CPU-only (gloo). An xccl barrier here deadlocks the run silently because trainer ranks become live L0 clients while vLLM TP init is posting `all_reduce`. This was the W18 failure mode.
- Do not pair W17/W19 with `colocate_sleep` — that's a different path (single-process) and the W17 kill assumes Ray.
- `TORCHTUNE_RAY_COLOCATE_DRAIN_S=0` reintroduces the wedge intermittently. Keep ≥ 2 s.

### 6.4 Validation envelope (what has actually been tested)

- **Model**: Qwen3-8B
- **Topology**: 1 node, 8 trainer tiles, vLLM TP=8 on the same 8 tiles, Ray head with `--num-gpus=12`
- **Batch**: G=4, batch_size=32, max_model_len=1024, max_num_seqs=32
- **Steps**: 3
- **Result**: 3/3 clean, ratios=1.0000, kl ∈ {0.0008, 0.0007, 0.0031}, no UR40, no banned:1
- **Log**: `experiments/colocate/ray_colo_logs/W19_kill_respawn_20260508_001441/run.log`
- **Launcher**: `experiments/colocate/run_qwen3_8b_colocate_ray_W19_kill_respawn_gloo_barrier.sh`

**Not validated**: larger models (32B+), larger G, multi-node, longer runs (200+ steps), HBM-pressure regimes that change the kill/respawn timing. If you change any of these, expect to re-tune `DRAIN_S` and re-confirm.

---

## 7. Open work

| # | Item | Owner | Priority |
|---|------|-------|----------|
| 1 | Implement option (a) — batched `collective_rpc` for ray-colocate wsync. Wire as `vllm_weight_sync_method: ray_collective_batched`. Expect wsync 88-97 s → 10-30 s. | unassigned | High — biggest near-term lever |
| 2 | Append W17/W18/W19 evidence to `UPSTREAM_FILING_DRAFT_l0_resource_pool.md` and decide whether to file. The W17 result is strong evidence for upstream — it reproduces the wedge AND demonstrates a clean release condition. | unassigned | Medium |
| 3 | Investigate why gen is 830 tok/s on Qwen3-8B TP=8 instead of expected 2-4k tok/s. `enforce_eager=True` is likely a factor; `gpu_memory_utilization=0.30` may be starving KV. | unassigned | Medium — affects all colocate paths, not just W19 |
| 4 | 200-step W19 stability soak. We have 3 steps. The wsync re-init load on Ray + L0 + driver every step may produce slow-burn issues we have not seen yet. | unassigned | Low until W19 has a real user |
| 5 | Document the wsync_method enumeration. `vllm_backend.py` currently selects `_sync_ray_colocate_weights` for `colocate_ray` mode without a method knob. If we add (1), we need a config selector. | unassigned | Bundled with #1 |

---

## 9. `colocate_sleep` — same-process vLLM (VALIDATED 2026-05-09, 3-step safe window)

### 9.1 Architecture

Each of the 12 XPU tiles runs its own TP=1 vLLM engine **in the same Python process** as the FSDP2 trainer. Between generation and training phases, `vLLM.sleep()` offloads weights to CPU and releases all GPU storage (weights + KV cache), freeing maximum memory for FSDP training. After training, `wake_up()` restores weights (with updated params synced from FSDP) and reallocates KV cache.

This sidesteps the L0 co-tenancy bug entirely — there is only one OS process per tile, so no concurrent L0 clients exist.

### 9.2 Critical fix: `num_gpu_blocks_override`

Tests 3-5 crashed with banned:1 PDE at step 1 generation. Root cause:

1. **L0 `mem_get_info` bug**: returns tile 0's stats for all tiles, creating bimodal free-memory readings across even/odd ranks.
2. **Bimodal KV cache**: vLLM auto-sizes KV cache from `gpu_memory_utilization × free_mem`. Odd ranks got ~25 GiB KV cache; even ranks got ~19 GiB. (Expected: uniform ~3 GiB.)
3. **Allocator trimming**: during FSDP training, the PyTorch caching allocator trimmed ~18 GiB of free-pool blocks back to L0 on odd ranks (their KV cache had reserved far more than needed).
4. **UR handle leak**: each `zeMemFree` leaks a UR handle. ~18 GiB of frees across many blocks exhausted the L0 resource pool.
5. **banned:1 PDE**: at wake_up, vLLM `storage.resize_(n)` requests blocks from L0 with a corrupted/exhausted pool → unmapped virtual memory access → GPU page fault.

**Fix**: `num_gpu_blocks_override=1536` in `_init_vllm_tp1()` (`vllm_backend.py`). Bypasses `gpu_memory_utilization` auto-sizing entirely. Formula: `batch_size × grpo_samples × blocks_per_seq × 1.5` where `blocks_per_seq = ceil(max_model_len / 16)`. For Qwen3-8B G=4 B=4 max_model_len=1024: `4 × 4 × 64 × 1.5 = 1536 blocks = ~3 GiB KV cache`. Uniform across all ranks → no trimming → no UR leak → no crash.

### 9.3 Validation results (Qwen3-8B G=4)

**Config**: `recipes/configs/dev/experimental/qwen3_8b_grpo_colocate_sleep_xpu.yaml`
**Launcher**: `experiments/colocate/run_qwen3_8b_colocate_sleep.sh [nsteps] [G]`

**test7 (3-step, 1.5× KV)**: 3/3 clean, exit=0
**test11 (7-step, 1.05× KV)**: 4/7 clean, banned:1 at step 4 generation

| Step | Total | Gen | GRPO | Rewards | Ratios | Reserved (Rank 0) |
|------|-------|-----|------|---------|--------|-------------------|
| 0 | 31.3s | 21.7s | 8.5s | 6.016 | 1.0000 | 49.00 GiB |
| 1 | 27.0s | 20.7s | 6.2s | 6.589 | 1.0000 | 55.45 GiB |
| 2 | 27.3s | 21.0s | 6.2s | 5.885 | 1.0000 | 55.45 GiB |
| 3 | 47.3s | 20.8s | 26.4s | 5.807 | 1.0000 | 55.45 GiB |
| 4 | CRASH | — | — | — | — | banned:1 PDE |

- Memory is FLAT after step 1 (AdamW state materialization) on most ranks
- Crash is NOT from pytorch-level OOM — reserved was stable at 55.45 GiB
- Crash is from L0 UR handle exhaustion after ~4 sleep/wake cycles
- Step 3 grpo=26.4s (4× normal) due to allocator consolidation, then stable

### 9.3.1 Step limit investigation (tests 8-11)

| Test | KV mult | Steps | Clean | Crash point | Root cause |
|------|---------|-------|-------|-------------|------------|
| test7 | 1.5× | 3 | 3/3 ✓ | — | — |
| test8 | 1.5× | 4 | 3/4 | step 3 BWD | UR handle exhaustion |
| test9 | 1.5× | 5 | 4/5 | step 4 BWD | UR handle exhaustion |
| test10 | KV-only | 5 | 1/5 | step 2 BWD | 15 GiB weights pinned raised baseline |
| test11 | 1.05× | 7 | 4/7 | step 4 gen | UR handle exhaustion |

**Mitigations tried and ruled out:**
- **Reduced KV cache** (1.05× mult → 4.72 GiB vs 6.75 GiB): stabilized pytorch reserved but did not extend crash horizon
- **KV-only sleep** (weights stay on GPU): higher memory baseline → crashed EARLIER (step 2)
- **`empty_cache()` defrag**: leaks UR handles proportional to FSDP units per call — makes it worse
- **`expandable_segments`**: not available on XPU (`PYTORCH_XPU_ALLOC_CONF` unsupported)

**Root cause**: each `resize_(0)` / `resize_(n)` cycle on the ~472 vLLM tensors triggers L0
`zeMemAlloc`/`zeMemFree` in the caching allocator. Each cycle leaks UR handles at the Level Zero
runtime layer. After ~4 full sleep/wake cycles, the pool is exhausted → banned:1 PDE. This is the
same L0 resource leak described in `docs/bugs/UPSTREAM_FILING_DRAFT_l0_resource_pool.md`, triggered
here by allocator segment operations rather than `empty_cache()`.

**Production workaround**: checkpoint/restart every 3 steps:
```bash
for epoch in $(seq 1 $TOTAL_EPOCHS); do
    torchrun ... --config ... num_steps=3 resume_from_checkpoint=True
done
```
Effective throughput: ~83s per effective step (3 × 27s training + ~170s restart, amortized).
Still **3× faster than W2** (247s/step).

### 9.4 Performance comparison

| Path | Steady-state | Effective (with restart) | Speedup vs W2 | wsync cost |
|------|--------------|--------------------------|---------------|------------|
| **colocate_sleep** | **~27 s/step** | **~83 s/step** (3-step + restart) | **3-9×** | ~3s (in-process `tensor.copy_()`) |
| W2 (8+4 process-isolated) | 247 s/step | 247 s/step | 1× | ~87s XCCL broadcast |
| W19 (kill+respawn) | ~290 s/step | ~290 s/step | 0.85× | ~125s (kill+respawn+RPC) |

The 9× steady-state improvement comes from three factors:
1. **12 tiles generate** (vs 4 on W2) → gen completes faster per step
2. **No cross-process wsync** — weight sync is `param.data.copy_()` against in-process buffers (~3s vs 87s)
3. **No Ray/RPC overhead** — everything is in-process

With periodic restart (3-step cycles), effective throughput drops to ~83s/step but is still 3× faster than W2.

### 9.5 Memory budget per tile (64 GiB)

| Phase | Breakdown | Total |
|-------|-----------|-------|
| Generation | vLLM model ~15.27 GiB (TP=1 full) + KV ~4.72 GiB (537 blocks) + FSDP shard 1.3 GiB | ~21 GiB |
| Training | FSDP shard 1.3 GiB + grad 1.3 GiB + AdamW 8 GiB + activations ~15 GiB | ~26 GiB |

Sleep/wake alternation ensures gen and train memory never overlap. Post-step reserved stabilizes at ~55.45 GiB after step 1 (8.55 GiB headroom).

### 9.6 Hard rules

- **Limit to 3 steps per process** — checkpoint/restart for longer training. L0 UR handle exhaustion causes banned:1 at step 4-5.
- **Always set `num_gpu_blocks_override`** — do NOT rely on `gpu_memory_utilization` auto-sizing. The L0 `mem_get_info` bug makes auto-sizing bimodal.
- **Always set `reshard_after_forward: true`** — ZeRO-2 combined with vLLM's weight copy pushes alloc to 55+ GiB.
- **Set `vllm_kv_cache_multiplier: 1.05`** — minimizes KV cache overhead (default 1.1 is safe; 1.5 is wasteful).
- **Never call `torch.xpu.empty_cache()`** in the training loop — it leaks UR handles proportional to FSDP unit count.
- **Do not attempt a "warm fence"** (`torch.empty(large)` to flush cached blocks before wake_up). On a fragmented allocator, this triggers internal L0 alloc/free cycles that leak UR handles.
- `colocate_sleep` is TP=1 only. For TP>1, use W2 (process-isolated) or W19 (kill+respawn).

### 9.7 Config and launch

```yaml
vllm_mode: "colocate_sleep"
vllm_gpu_memory_utilization: 0.40      # Affects prefill budget, NOT KV sizing
vllm_max_model_len: 512                # Keep ≤1536 (plen>2048 crashes)
vllm_kv_cache_multiplier: 1.05        # Minimize KV cache overhead
reshard_after_forward: true            # CRITICAL — ZeRO-2 OOMs with colocate
num_steps: 3                           # Max 3 steps per process (L0 UR limit)
# vllm_num_gpu_blocks_override: 537    # Auto-calculated if omitted
```

```bash
bash experiments/colocate/run_qwen3_8b_colocate_sleep.sh [nsteps] [G]
```

---

## 10. Cross-references

- Mechanism (UR40 deep dive): `docs/bugs/xpu_l0_event_pool_co_tenancy.md`
- Upstream draft (frozen pre-W17): `docs/bugs/UPSTREAM_FILING_DRAFT_l0_resource_pool.md`
- Phase 1/2 elimination: `docs/reports/colocate_tp8_status_20260506.md`
- Phase 4 Ray status (frozen at W2): `docs/reports/colocate_ray_tp8_status_20260507.md`
- Weight sync architecture (general): `docs/features/vllm_weight_sync.md`
- Ray-on-Aurora setup notes: `docs/features/ray.md`
- Recipe: `recipes/dev/grpo_full_finetune_distributed_xpu.py` (`vllm_mode=colocate_ray` block in `train()`)
- vLLM backend init: `torchtune/dev/rl/vllm_backend.py:_init_vllm_ray_colocate` (line 907) and `_setup_vllm_ray_colocate_mode` (line 983)
- Wsync runtime: `torchtune/dev/rl/weight_sync.py:_sync_ray_colocate_weights` (line 128)
- colocate_sleep vLLM init: `torchtune/dev/rl/vllm_backend.py:_init_vllm_tp1` (num_gpu_blocks_override)
- colocate_sleep config: `recipes/configs/dev/experimental/qwen3_8b_grpo_colocate_sleep_xpu.yaml`
- colocate_sleep launcher: `experiments/colocate/run_qwen3_8b_colocate_sleep.sh`
