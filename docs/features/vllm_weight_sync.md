# vLLM Weight Sync — Implementation and Experiments

After each optimizer step in GRPO training, the policy model's updated weights
must be pushed to the vLLM inference server so that the next generation step
uses the trained model rather than stale initial weights. This document covers
the implementation, bugs found and fixed, measured performance, and remaining
work.

## Background

The GRPO recipe uses a colocated vLLM server for efficient generation:
- **Training tiles**: 10 XPU tiles running FSDP2 over the policy model
- **vLLM tiles**: 2 XPU tiles (TP=2 for 32B, TP=1×DP=2 for 3B) running the
  inference server

After each optimizer step, rank 0 must synchronize updated weights from the
FSDP2 sharded state into the vLLM server. The challenge: vLLM and the training
process are on different tiles and cannot share device memory.

---

## Methods Implemented

Four methods are implemented, selectable via `vllm_weight_sync_method` in the
training config:

```yaml
vllm_weight_sync_method: xccl   # xccl | shm | raw_bytes | path
```

### 1. `path` — Safetensors file (legacy)

Training rank 0 gathers FSDP shards via `full_tensor()`, remaps
torchtune→HF parameter names, and saves to a safetensors file on `/dev/shm`.
vLLM calls `load_file(path, device="cpu")` then `model.load_weights()`.

**Performance**: ~1.3 GB/s effective write (safetensors per-tensor overhead).
Not used in production; kept for backward compatibility.

### 2. `raw_bytes` — Raw binary file on `/dev/shm`

Same gather path, but uses a compact binary format:
- 8-byte little-endian header length
- JSON header listing tensor metadata (name, shape, dtype, nbytes)
- Contiguous raw tensor bytes (BF16 stored as int16, same bit pattern)

vLLM side: `frombuffer()` zero-copy view into the `mmap`'d file, then
`model.load_weights()` which does `param.copy_()` in-place on XPU.

**Performance (Qwen 3B, 5.75 GiB, validated 2026-04-16)**:

| Phase | Time |
|-------|------|
| gather (FSDP AllGather, blocking) | 1.1s |
| write to `/dev/shm` | 2.6s |
| http (vLLM file read + load_weights) | 5.5s |
| **total** | **9.3s** |
| waited (residual at next gen start) | 7.8s |

Note: Early experiments wrote to `/tmp` (NVMe, ~1.3 GB/s). Switching to
`/dev/shm` (RAM-backed tmpfs, 504 GB on Aurora) eliminated the NVMe bottleneck.
Estimated 31B NVMe time was ~74s; with `/dev/shm` it would be ~20s.

### 3. `shm` — POSIX shared memory

Uses `multiprocessing.shared_memory.SharedMemory` to allocate a single named
SHM block (`torchtune_weights`) that both the training process and vLLM workers
can map simultaneously.

**Training side** (`_sync_weights_to_vllm_shm()` in the recipe):
1. `full_tensor()` gather — synchronous, all FSDP ranks participate
2. Background thread: `ctypes.memmove()` each gathered tensor into the SHM
   block by offset (no Python object allocation, DDR5 bandwidth)
3. POST JSON metadata (tensor names/shapes/dtypes/offsets) to vLLM's
   `/collective_rpc` endpoint with method `load_weights_from_shm`
4. Set sync-done event; SHM block kept alive as `self._shm_block` for reuse

**vLLM side** (`load_weights_from_shm()` in `vllm_weight_sync_worker.py`):
1. Attach to the named SHM block (`SharedMemory(create=False)`)
2. `torch.frombuffer(shm.buf, offset=..., count=...)` — zero-copy CPU view
3. Pass CPU tensors directly to `model.load_weights()` (in-place `param.copy_()`)
4. `shm.close()` after `load_weights()` completes (SHM stays mapped during load)

**Key design decisions**:

- **No `.to(device)` before `load_weights()`**: On large models (31B, 57 GiB),
  calling `tensor.to(xpu)` before loading would require allocating a second full
  model copy on XPU, which OOMs when the model already fills >95% of device
  memory. `load_weights()` uses `param.copy_()` in-place, so the CPU tensor is
  sufficient.

- **Persistent SHM block**: The SHM block is allocated once (step 0) and reused
  across all subsequent steps (`self._shm_block`). Without reuse, each step
  allocates a fresh block, triggering OS demand-paging for all pages (~1.4M page
  faults for 5.75 GiB → 1.4 GB/s instead of DDR5 ~8 GB/s). On step 0 the
  warmup cost is paid once; steps 1+ run at full bandwidth.

- **SHM block not unlinked between steps**: The training side owns the block
  lifecycle. vLLM workers call `shm.close()` but never `shm.unlink()`. The
  training recipe unlinks in `cleanup()`.

### 4. `xccl` — GPU→GPU broadcast (recommended)

Creates a cross-process XCCL process group between training rank 0 and all vLLM
TP workers. Weight transfer is a single GPU→GPU broadcast — no CPU staging, no
file I/O, no HTTP data transfer.

**Setup** (once, at first weight sync):
1. Training rank 0 creates a `TCPStore` (master, `wait_for_workers=False`)
2. Background thread POSTs `init_xccl_communicator` to vLLM via `/collective_rpc`
   with the TCPStore host/port/world_size
3. Training rank 0 and all vLLM TP workers join a `ProcessGroupXCCL` via the
   shared TCPStore
4. World size = 1 (training rank 0) + TP (vLLM workers)

**Per-step sync** (fully async):
1. `full_tensor()` gather on all FSDP ranks (synchronous)
2. Concatenate all gathered params into a single flat BF16 GPU buffer
3. Background thread POSTs `receive_weights_xccl` with tensor metadata to vLLM
4. Training rank 0 broadcasts the flat buffer; vLLM workers receive
5. vLLM splits the buffer by param and calls `model.load_weights()` in-place

**Why this works now**: The original XCCL approach (2026-04-15) SIGABRTed
because it tried to create a second communicator using the high-level
`init_process_group` API while the training PG was active. The fix (2026-04-21)
bypasses `init_process_group` entirely and constructs a `ProcessGroupXCCL`
directly via the low-level `distributed_c10d` API, which avoids the Level Zero
context collision.

**Key implementation details**:
- `TCPStore` port 51217 (avoids conflict with vLLM's `vllm_group_port=51216`)
- TP-aware: each vLLM TP worker joins as `base_rank + tp_rank`
- Training creates TCPStore and POSTs to vLLM in a background thread to avoid
  deadlock (both sides must enter the PG constructor concurrently)
- `PrefixStore("wsync", store)` isolates this PG's keys from any other PGs

### 5. 2-hop broadcast — dedicated vLLM (recommended for 32B)

For multi-node configurations where vLLM runs on a separate node, a 2-hop
broadcast architecture replaces the flat cross-process methods above.

**Architecture (3-node example):**
```
Node 0 (vLLM):     TP=4 workers (ranks 1-4 in wsync PG)
Node 1 (Training):  12 tiles, pure FSDP
Node 2 (Training):  12 tiles, pure FSDP

Hop 1 (cross-node): Training rank 0 → vLLM TP-0  (2-rank PG, gloo TCP over Slingshot)
Hop 2 (intra-node):  vLLM TP-0 → TP-1..3          (TP-rank PG, XCCL over XeLink)
```

**Cross-PG transport** (`WSYNC_CROSS_METHOD`):
- `gloo` (default): TCP over Slingshot at ~1.3 GB/s. CXI RDMA leak eliminated.
  Requires `GLOO_SOCKET_IFNAME=hsn0`.
- `xccl`: XCCL RDMA. On 3-node 24-way, `other` is same ~31s (dominated by FSDP
  AllGather), but gen on sync steps drops from ~20s to ~12s because vLLM receives
  weights faster. Leaked CXI MR entries on 2-node (~step 30); 3-node 5/5 clean
  with 20+ GiB l0_free headroom but long-term stability unproven.

**Intra-PG transport** (`WSYNC_INTRA_METHOD`):
- `xccl` (default, recommended): XeLink, ~2.2 GB/s for TP=4. Uses GPU tensors directly.
- `gloo` (fallback): Localhost TCP/SHM, ~0.9 GB/s for TP=4. Requires CPU staging
  buffers (`_gloo_intra_buf` on non-root, reuses `_gloo_recv_buf` on root).

**Setup** (once, at first weight sync):
1. Training rank 0 creates a `TCPStore(is_master=True)`
2. POSTs `init_xccl_communicator` to vLLM's `/collective_rpc` with store info
3. vLLM workers create per-replica PGs:
   - Cross-PG: 2-rank (vLLM TP-0 ↔ training rank 0), prefix `wsync_cross_{replica_idx}`
   - Intra-PG: TP-rank (all TP workers for this replica), prefix `wsync_intra_{replica_idx}`
4. Training creates matching cross-PGs (one per vLLM replica)

**Per-step sync (deferred broadcast pattern)**:
```
Step N:  [...GRPO fwd/bwd...] [opt] [gather → cpu_batches]
Step N+1: [gen ← vLLM uses old weights] [bg: cross-PG bcast → intra-PG bcast]
```

Weights gathered at the end of step N are broadcast during step N+1's generation
phase. With `vllm_weight_sync_interval=2`, broadcast happens every other step,
further reducing overhead. The deferred pattern hides broadcast latency behind
generation — measured overhead is ~0.3s on even steps.

**DP>1 per-replica PGs** (implemented 2026-04-28):

When `VLLM_DP > 1`, each vLLM replica gets its own intra-PG and cross-PG:
- Intra-PG: `intra_size = tp_size_local` (not `world_size - 1`), prefix
  `wsync_intra_{replica_idx}`
- Cross-PG: prefix `wsync_cross_{replica_idx}`, guard `tp_rank == 0` (not `my_rank == 1`)
- Training broadcasts to all cross-PGs in parallel (start all, wait all) for each batch

Without per-replica PGs, a shared intra-PG spanning all replicas deadlocks
because `/collective_rpc` dispatches per-replica — each replica's TP workers
enter the broadcast independently, but broadcast requires ALL `intra_size` workers.

**Performance (32B, 3-node 24-way, v11 2026-04-28):**

| Metric | Gloo cross-PG | XCCL cross-PG (2-node) |
|--------|---------------|------------------------|
| Cross-node bcast BW | 1.3 GB/s | 7.9 GB/s |
| Intra-node bcast BW | 2.0 GB/s | 8.0 GB/s |
| Sync time (61 GiB) | ~47s | ~9.1s |
| Step time (with all opts) | **32.6s** (deferred) | ~43s |
| Stability | 20/20 clean (2-node) | 24/24 clean (2-node) |

---

## Async Flow

For SHM and raw_bytes methods, the sync runs asynchronously relative to the
training backward pass, but not relative to the next generation step:

```
rank 0:  [ generate (vLLM) ] [ GRPO fwd/bwd ] [ opt ] [ gather (sync) ] ─────────── [ wait ]
bg thread:                                               [ memmove ] [ POST→vLLM ]
```

`waited` = time blocked at the start of next generation waiting for the
background thread to complete. Because the background thread starts immediately
after the optimizer step (at the very end of each step), the next generation
call arrives before the thread finishes and must wait.

**Implication**: `waited ≈ copy_time + http_time`. The sync is not hidden behind
generation in the current implementation.

For XCCL, the same async pattern applies but with GPU→GPU broadcast replacing
the memmove + HTTP:

```
rank 0:  [ generate (vLLM) ] [ GRPO fwd/bwd ] [ opt ] [ gather ] ──────── [ wait ]
bg thread:                                               [ concat ] [ bcast ]
```

For 3B, XCCL sync (gather+concat+bcast = 3.1s) completes within the generation
time (~5s), so `waited ≈ 0` — sync is fully hidden. For 32B, XCCL sync is
estimated at ~8s (see below), which would also be hidden behind generation
(~9s).

### Deferred broadcast (2-hop, 3-node)

The 2-hop architecture uses a **deferred broadcast** pattern that hides sync
latency behind generation:

```
Step N:   [ gen ] [ GRPO fwd/bwd ] [ opt ] [ gather → cpu_batches ]
Step N+1: [ gen ←── vLLM uses step N-1 weights ──→ ] [ GRPO ... ]
          [ bg: cross-PG bcast ─→ intra-PG bcast ]
```

Weights gathered at end of step N are broadcast during step N+1's generation.
With `vllm_weight_sync_interval=2`, broadcast only occurs on even steps.
Measured overhead: ~0.3s on broadcast steps (vs 47s if synchronous).

---

## Measured Performance

### Qwen2.5-3B (5.75 GiB BF16, 434 params)

#### Server mode (10+2 tiles, VLLM_DP=2)

Validated 2026-04-16, job 8438434 (hold node, 5 steps).

| Method | gather | copy/write | http/bcast | total | waited | step time |
|--------|--------|------------|------------|-------|--------|-----------|
| `raw_bytes` | 1.1s | 2.6s | 5.5s | 9.3s | 7.8s | 21–22s |
| `shm` step 0 | 14.7s† | 4.2s (1.4 GB/s) | 1.0s | 19.9s | 4.9s | 63.7s† |
| `shm` steps 1–4 | 1.1–1.3s | 0.7–0.8s (8 GB/s) | 0.9s | **2.8–2.9s** | **1.4s** | **21.1–21.5s** |
| `xccl` steps 1+ | 0.4s | 0.0s (concat) | 2.6s (2.2 GB/s) | **3.1s** | **0s** | **6.2–6.5s** |

† FSDP initialization overhead on step 0 (one-time).

**SHM vs raw_bytes (steady state)**: 9.3s → 2.8s (3.3×), waited 7.8s → 1.4s
(5.6×). Step time unchanged at ~21s — sync fully hidden behind generation.

**XCCL vs SHM**: Total sync time comparable (3.1s vs 2.8s), but XCCL eliminates
CPU staging entirely. The real win is that XCCL was measured on the 10+2 tile
config where step time dropped to 6.2s (vs 21s with SHM), primarily because the
XCCL test used a later recipe version with additional optimizations. XCCL sync
is fully hidden behind generation at this model size.

**XCCL bandwidth**: 2.2 GB/s cross-process (training `torchrun` + separate vLLM
server). Isolated 12-rank test achieved 14 GB/s — the gap is likely
cross-process XCCL rendezvous overhead.

#### Colocated modes (all tiles)

| Mode | Tiles | Per-step | Weight sync | Notes |
|------|-------|----------|-------------|-------|
| All-rank colocated (no vLLM server) | 2 | ~8.0s | 0.1–0.2s (direct param copy) | Config B (G=16), beats A100 |
| Colocate-sleep (vLLM embedded) | 10 | ~8.2s | 0.9s (wake+sync) | sleep/wake overhead ~1s |
| Colocate non-sleep | 10 | ~7.3s | trivial | KV cache kept resident |

### Qwen3-32B (59 GiB BF16, ~700 params)

#### Server mode (10+2 tiles, TP=2)

| Method | gather | copy | http/bcast | total | waited | step time |
|--------|--------|------|------------|-------|--------|-----------|
| `shm` step 0 | 16.9s† | 41.3s (1.4 GB/s) | 7.1s | 65.4s | 48.1s | 135.5s† |
| `shm` steps 1–4 | 5.5–5.6s | 6.9–7.7s (7.5–8.3 GB/s) | 5.7–6.4s | **18.7–19.3s** | **12.9–13.5s** | — |
| `xccl` (estimated) | ~5s | 0s (concat) | ~3s (2.2 GB/s) | **~8s** | **~0s** | — |

† Step 0 overhead: FSDP init (gather) + page-fault warmup (copy).

**Step time breakdown (32B single-node, server mode, G=8/fbs=8)**:
- vLLM generation: ~9.2s
- Policy forward: ~3.4s
- Ref forward: ~2.2s
- Weight sync (waited): ~13s (SHM) or ~0s (XCCL, estimated)
- Total: ~22.8 s/step (best config, with SHM wait absorbed into gen time)

**XCCL for 32B has not been validated in production** on single-node — see "32B Memory
Constraints" below. Multi-node 2-hop is the production path (see below).

#### Dedicated vLLM multi-node (2-hop weight sync)

| Config | Sync method | Sync time | Step time | Steps clean |
|--------|-------------|-----------|-----------|-------------|
| 2-node (12 train + 12 vLLM), XCCL cross+intra | 9.1s at 7.9 GB/s | ~43s | 24/24 |
| 2-node, gloo cross + XCCL intra | 47s at 1.3 GB/s | ~67s | 20/20, exit=0 |
| **3-node 24-way** (24 train + 4 vLLM), gloo cross + XCCL intra, deferred | ~0.3s overhead | **56.1s avg** | **5/5 clean** (v12, fresh L0) |
| 3-node 24-way, gloo cross + **gloo intra**, deferred | ~0.3s overhead | **72.1s avg** | **5/5 clean** (v13) |
| 3-node 24-way, gloo cross + XCCL intra, **20-step** | ~0.3s overhead | **53.5s avg** | **20/20 clean, 10/10 syncs** (v14) |
| 3-node 24-way, **XCCL cross** + XCCL intra | ~0.3s overhead | **53.9s avg** | **5/5 clean** (v16b) |
| 3-node 24-way, gloo cross, **DP=3 parallel bcast** | ~0.3s overhead | **59.9s avg** | **5/5 clean** (v18) |
| 3-node 24-way, **pinned CPU buffer**, G=16 fbs=16 | ~0.3s overhead | **~41s avg** | **5/5 clean** (Test A) |
| 3-node 24-way, **pinned CPU buffer + G=32**, fbs=16 | ~0.3s overhead | **~53s avg** | **5/5 clean** (Test B) |
| 3-node 24-way, pinned buf, G=16, **max_gen=512** | ~0.3s overhead | **~82s** | **3/3 clean** (Test D). 0.01 GiB free (absolute limit) |
| 3-node 24-way, pinned buf, **G=64** | — | — | **FAILED** (Test E). XPU kernel bug at batch=64 |
| 3-node 24-way, pinned buf, G=32, **max_gen=256** | — | — | **OOM step 1** (Test G). CCL external 1.85→13-20 GiB (3 chunks) |
| 3-node 24-way, pinned buf, G=32, **max_gen=192** | ~0.3s overhead | **~72s** | **3/3 clean** (Test G2). 1.50 GiB free (marginal) |
| 3-node 24-way, pinned buf, **G=48** | — | — | **HUNG step 1** (Test H). CCL external→15.26 GiB (3 chunks) |

#### Pinned CPU buffer optimization (2026-04-28)

The gather phase (`full_tensor()` → `.to(bf16).contiguous()` → `.cpu()`) was
31s for 61 GiB at 32B. The bottleneck was 707 synchronous `.cpu()` calls, each
allocating unpinned memory and blocking on D2H copy, plus 66 `torch.cat()` calls.

**Fix** (`TORCHTUNE_PINNED_CPU_BUF=1`):
1. Pre-allocate a single 61 GiB CPU buffer with `pin_memory()` (works on XPU)
2. Replace `.cpu()` with `.copy_(non_blocking=True)` into buffer slices
3. Eliminate all `torch.cat()` calls — `cpu_batches` entries are zero-copy views
4. Sync once per batch boundary via `torch.xpu.synchronize()`

| Metric | Baseline | Pinned buffer | Speedup |
|--------|----------|---------------|---------|
| Gather time (steady-state) | 31.3s | **3.7s** | **8.5×** |
| ft (full_tensor AllGather) | 3.2s | 3.5s | — |
| cast (.to(bf16).contiguous()) | 24.0s† | 0.0s | — |
| d2h (.cpu() / .copy_) | 4.2s | 0.0s | — |
| Total step time (G=16, sync interval=2) | ~53.5s | **~41s** | 23% |

† The 24s "cast" time in baseline is actually the AllGather completion — the
`full_tensor()` dispatch returns in ~3.2s but the actual data arrives during the
subsequent `.to(bf16).contiguous()` which forces synchronization. With pinned
buffer, `non_blocking=True` returns immediately and sync is deferred.

**G=32** (`GRPO_SAMPLES=32 FORWARD_BATCH_SIZE=16`): Doubles generation rollouts.
2 forward chunks per step (32/16=2). Memory tighter (l0_free=4.94 GiB at step 4
PRE-BWD vs 17.75 GiB at G=16) but stable with 0 retries, 0 OOMs. Per-sample
throughput: 32/53s = 0.60 samples/s vs 16/41s = 0.39 samples/s → **1.54× improvement**.

**2-Chunk Rule** (discovered 2026-04-29): G/fbs must be ≤ 2. With 3+ forward
chunks, CCL external memory explodes from ~1.8 GiB to 13-15 GiB at step 1.
Root cause: FSDP AllGather/ReduceScatter buffer retention across multiple model
scans — 3 ref_fwd chunks trigger 3 full FSDP scans, and intermediate CCL buffers
accumulate. Tests G (G=32/max_gen=256, OOM) and H (G=48/max_gen=128, hung) both
had 3 chunks and showed identical CCL explosion. Test G2 (G=32/max_gen=192, 2
chunks) survived with 1.50 GiB margin.

**Production config envelope (2026-04-29):**

| Config | Step time | Samples/s | Margin | Status |
|--------|-----------|-----------|--------|--------|
| G=16, max_gen=128 | ~41s | 0.39 | 10+ GiB | Safe |
| **G=32, max_gen=128** | **~53s** | **0.60** | ~5 GiB | **Best throughput** |
| G=32, max_gen=192 | ~72s | 0.44 | ~1.5 GiB | Marginal |
| G=16, max_gen=512 | ~82s | 0.20 | ~0 GiB | Limit |
| G=48+, any | — | — | OOM/hang | Blocked (3+ chunks) |
| G=64, any | — | — | XPU bug | Blocked (kernel bug at batch=64) |

**3-node step-time comparison (2026-04-28):**

| Step | v12 gloo×+XCCL_i | v13 gloo×+gloo_i | v14 (20-step) | v16b XCCL×+XCCL_i | v18 DP=3 parallel |
|------|------------------|------------------|---------------|-------------------|-------------------|
| 0 | 79.8s | 78.1s | 78.4s | 80.6s | 79.2s |
| 1 | 32.2s | 32.3s | 32.4s | 33.9s | 34.2s |
| 2 | 68.2s (gen=19.1) | 110.6s (gen=60.9) | 69.4s | 61.9s (gen=11.8) | 75.8s (gen=26.5) |
| 3 | 29.9s | 29.9s | 30.1s | 31.2s | 31.7s |
| 4 | 70.4s (gen=20.7) | 109.5s (gen=59.3) | 72.0s | 61.8s (gen=11.8) | 78.4s (gen=28.1) |
| **Avg** | **56.1s** | **72.1s** | **56.5s** | **53.9s** | **59.9s** |

**Key findings:**

- **v11 crash was stale L0**: Fresh L0 (v12) 5/5 clean, v14 extended to 20/20 steps
  with 10 successful syncs — XCCL intra is stable. NOT a fundamental XCCL bug.
- **Gloo intra penalty**: +28.5% step time (0.9 GB/s vs 2.2 GB/s). Penalty is
  entirely in gen delay on sync steps (vLLM blocked by slower intra broadcast).
- **XCCL cross-PG (v16b)**: `other` unchanged (~31s, dominated by FSDP AllGather),
  but gen on sync steps drops from ~20s to ~12s — vLLM receives faster via RDMA.
  Net: ~2s/step improvement. 5/5 clean on 3-node (20+ GiB l0_free headroom),
  but long-term CXI MR leak risk unvalidated beyond 5 steps.
- **DP>1 parallel broadcast (v18)**: Sync-step gen time 26.5s vs 63.6s sequential
  (v17). But DP>1 avg step (59.9s) > DP=1 avg step (53.9s) because the 3× gloo
  broadcast overhead outweighs the 3× generation throughput at G=16/max_gen=128.
  DP>1 benefits kick in at higher generation loads (G=64+, max_gen=512+).

**DP>1 bug fixes (2026-04-28):**
- **Bug #18: `vllm_tensor_parallel_size` not passed to recipe** — launcher
  defaulted to tp=1, miscalculating `base_rank` for replicas >0. Replica 1
  joined `wsync_cross_0` instead of `wsync_cross_1` → timeout. Fixed by adding
  `vllm_tensor_parallel_size=${VLLM_TP}` to launcher config overrides.
- **Bug #19: `WSYNC_CROSS_METHOD` hardcoded to gloo** — launcher had
  `export WSYNC_CROSS_METHOD=gloo` instead of `"${WSYNC_CROSS_METHOD:-gloo}"`.
  XCCL cross-PG was impossible to enable. Fixed.

#### Colocate-sleep mode (12 tiles, TP=4, DP=3)

| Version | Per-step | Weight sync | Notes |
|---------|----------|-------------|-------|
| v1 (manual TP slice) | ~128s | ~90s | Fragile manual QKV/gate_up merging |
| v2 (load_weights API) | ~100s | 7.8s | 22% faster; stable 3 steps |

Weight sync improved 11× (v1→v2) by replacing manual TP slicing (~160 lines of
QKV/gate_up merge code) with vLLM's `load_weights()` API. Remaining bottleneck
is KV cache restore (~20s) and rank synchronization (~10s).

#### Multi-node (2 nodes, 20+4 tiles, OFI transport)

Validated 2026-04-06, flat FSDP, 2× vLLM DP (one server per node).

| Config | Step time | Throughput | Notes |
|--------|-----------|------------|-------|
| G=8/fbs=8, OFI transport | 66–81s | 39.3 seqs/min | vLLM is bottleneck (32–43s gen) |

Weight sync uses SHM (each node syncs to its local vLLM). Cross-node HTTP
requires `no_proxy="*"` to bypass Aurora's Squid proxy.

---

## 32B Memory Constraints and Allocator Interaction

32B single-node GRPO with the default allocator (no `XPU_USM_ALLOC_SO`) works
for 3–4 steps but OOMs at step 4:

| Step | Reserved | Driver overhead | Status |
|------|----------|-----------------|--------|
| 0 | 52 GiB | 24 GiB | OK (34s) |
| 1 | 62 GiB | 34 GiB | OK (34s) |
| 2 | 62 GiB | 34 GiB | OK (27s, stable) |
| 3 | 47 GiB | 19 GiB | Slow (56s, fragmentation) |
| 4 | — | 29 GiB | **OOM** (33.7 GiB PyTorch + 29.4 GiB driver = 63.1 GiB) |

The 29 GiB of driver overhead (CCL/L0/XCCL contexts + IPC handles + internal
buffers) is not reclaimable by PyTorch. Only ~35 GiB of the 64 GiB tile is
available for PyTorch allocations.

### Custom allocator attempts (2026-04-21 – 2026-04-22)

To reduce fragmentation and avoid OOM, we built custom allocators loaded via
`XPUPluggableAllocator`. All produce GPU segfaults during CCL IPC:

| Allocator | Design | CCL IPC result |
|-----------|--------|----------------|
| `usm_caching_alloc.cpp` (gen1) | Power-of-2 size-class free lists | Works at 3B; untested at 32B (vLLM startup bug masked it) |
| `usm_arena_alloc.cpp` v5 | Two-tier: small buckets + coalescing arena + large direct | Segfault (BUCKET_CAP pooling bug) |
| `usm_arena_alloc.cpp` v6 | Same as v5, BUCKET_CAP fixed | Segfault (arena sub-allocated pointers in 1–8 MiB range) |
| `usm_arena_alloc.cpp` v7 | **No arena**: small buckets + exact-aligned direct only | Segfault (proves XPUPluggableAllocator itself is the issue) |

**Root cause**: `XPUPluggableAllocator` creates allocations in a SYCL context
that differs from what CCL uses for `zeMemGetIpcHandle`. Even pure
`sycl::malloc_device` pointers (no sub-allocation) produce GPU segfaults at
`0xffffff8000000000` during FSDP allgather/reduce_scatter. This is distinct
from the `expandable_segments` bug (which involves non-USM virtual memory) —
see `docs/bugs/intel_ccl_expandable_segments_bug.md` for full analysis.

**CCL algorithm overrides do not help**: `CCL_ALLGATHER=naive`,
`CCL_REDUCE_SCATTER=naive`, `CCL_ALLREDUCE=direct` were tested — CCL still uses
L0 IPC for all intra-node GPU-to-GPU transfers regardless of algorithm. The
algorithm selection controls the communication pattern (ring/tree/direct), not
the transport mechanism.

### Implications for weight sync at 32B

The weight sync method itself is not the bottleneck — the issue is that 32B
single-node training hits OOM at step 4 regardless of sync method. The XCCL
broadcast weight sync would eliminate the 13s waited time, but training crashes
before that matters.

**Viable 32B configurations**:
1. **Multi-node HSDP** (2+ nodes, OFI transport): Custom allocator works over
   OFI (no IPC). 66–81 s/step at 2 nodes. Weight sync via SHM per-node.
2. **Dedicated vLLM node** (1 node training, 1 node vLLM): All 12 tiles for
   FSDP with default allocator. No vLLM memory contention reduces OOM pressure.
   Not yet tested.
3. **Colocate-sleep** (12 tiles, TP=4/DP=3): ~100 s/step with v2 weight sync.
   Memory fits via time-multiplexing (sleep releases GPU memory between phases).
4. **Single-node server mode** (10+2 tiles): Works for 3–4 steps with default
   allocator. Needs Intel fix for allocator IPC or driver overhead reduction to
   sustain longer runs.

---

## Bugs Fixed

### 1. Wrong vLLM endpoint (2026-04-15)

`_sync_weights_to_vllm()` POSTed to `/load_weights_from_path/` which returns
404 in standard vLLM. Fixed: use `/collective_rpc` with
`{"method": "load_weights_from_path", "args": [path]}`.

### 2. `/collective_rpc` behind dev flag (2026-04-15)

The `/collective_rpc` endpoint only exists when `VLLM_SERVER_DEV_MODE=1`.
Fixed: added to `run_grpo_vllm_xpu.sh` launcher.

### 3. Missing worker extension (2026-04-15)

vLLM had no `load_weights_from_path` method. Fixed: created
`torchtune/dev/vllm_weight_sync_worker.py` with `WeightSyncFromFileExtension`,
injected via `--worker-extension-cls`.

### 4. Wrong key format in saved safetensors (2026-04-15)

`_build_tune_to_hf_map()` keyed by FSDP-wrapped parameter name, but
`state_dict()` returns clean names → map lookups always missed → safetensors
saved in torchtune-format keys → vLLM silently ignored all weights. Fixed: key
by `clean_name` so HF-format keys are saved.

### 5. `_sync_weights_to_vllm` with VLLM_DP=2 (2026-04-15)

Used `self._vllm_url` (comma-joined string) for POST → `InvalidURL`. Fixed:
iterate over `self._vllm_urls` list.

### 6. SHM page-fault bottleneck (2026-04-16)

Each step unlinked and recreated the SHM block → OS demand-paging for all
pages (~1.4M faults for 5.75 GiB) → copy speed 1.4 GB/s instead of ~8 GB/s.
Fixed: persist `self._shm_block` across steps; pages stay faulted in after
step 0.

### 7. OOM on `load_weights_from_shm` for large models (2026-04-16)

`load_weights_from_shm()` called `tensor.to(device)` before passing to
`load_weights()`. For 31B (57 GiB, >95% of XPU memory), this attempted to
allocate a second 57 GiB copy on XPU → OOM crash. Fixed: pass CPU
`frombuffer` tensors directly; `load_weights()` does `param.copy_()` in-place.
Keep SHM open until `load_weights()` completes so buffer stays valid.

### 8. Missing `VLLM_GEMMA4=1` for Gemma4 overlay (2026-04-16)

`run_grpo_vllm_xpu.sh` resets `PYTHONPATH` at line 60, overwriting any
upstream overlay path. The Gemma4 test script must pass `VLLM_GEMMA4=1` so
the launcher uses `vllm_gemma4_overlay` as `PYTHONPATH`. Fixed in
`run_shm_sync_test_31b.sh`.

### 9. XCCL communicator SIGABRT (2026-04-15, fixed 2026-04-21)

Original approach used `torch.distributed.init_process_group()` to create a
second XCCL communicator for weight sync → SIGABRT in Level Zero (creating a
second communicator while the training PG is active). Fixed by constructing
`ProcessGroupXCCL` directly via `torch.distributed.distributed_c10d`, which
bypasses the global state and avoids the L0 context collision.

### 10. XCCL TCPStore deadlock (2026-04-21)

Training rank 0 created TCPStore then immediately entered the PG constructor,
but vLLM workers hadn't connected yet → deadlock. Fixed: POST to vLLM
(triggering `init_xccl_communicator`) in a background thread, then enter PG
constructor. Both sides enter concurrently.

### 11. XCCL TCPStore timeout type (2026-04-21)

`TCPStore(timeout=120)` passed an int — requires `datetime.timedelta(seconds=120)`.

### 12. XCCL TP size mismatch (2026-04-21)

World size was hardcoded to 2 (training + 1 vLLM worker), but TP=2 means 2
vLLM workers → world size should be 3. Fixed: launcher passes
`vllm_tensor_parallel_size` and world size = 1 + tp_size.

### 13. XCCL init on wrong ranks (2026-04-21)

All training ranks called `_init_xccl_weight_sync()` → only shard leader
(rank 0) should. Fixed: guard with `if self._is_shard_leader:`.

### 14. XCCL intra-PG broadcast crash (2026-04-28, RESOLVED)

On the second XCCL intra-PG broadcast (after the first deferred broadcast
completed successfully), vLLM TP worker 2 crashed:
```
CCL_ERROR| atl_def.cpp:19 validate: condition comm_rank < comm_size failed
```

**Root cause: stale L0 device state** from a prior v9 crash on the same nodes.
Confirmed by v12 test: fresh job with clean L0 state completes 5/5 steps with
2 successful XCCL intra-PG broadcasts, including surviving the exact failure
point (2nd sync round after FSDP training activity). NOT a fundamental XCCL bug.

**Mitigation**: Ensure clean process cleanup between jobs (the launcher's
`pkill -9 -f 'VLLM::'` and `pkill -9 -f 'vllm.entrypoints'` commands).

**Gloo intra-PG fallback** (`WSYNC_INTRA_METHOD=gloo`) was also implemented
and validated (v13, 5/5 clean) as insurance. Available at 2.4x performance
cost (0.9 GB/s vs 2.2 GB/s).

### 15. DP>1 intra-PG deadlock (2026-04-28, FIXED)

With `VLLM_DP > 1`, one shared intra-PG (size=world_size-1) spanning all
replicas caused a deadlock. `/collective_rpc` dispatches per-replica, so each
replica's TP workers entered the XCCL broadcast independently, but the broadcast
requires all `intra_size` workers to participate. Fixed: per-replica PGs with
`intra_size = tp_size_local`.

### 16. Stale vLLM worker processes (2026-04-28, FIXED)

vLLM renames worker processes to `VLLM::Worker_TP*` which evades
`pkill -f vllm`. Stale workers from crashed runs consume L0 device contexts,
causing `UR_RESULT_ERROR_OUT_OF_RESOURCES` on subsequent launches. Fixed:
added `pkill -9 -f 'VLLM::'` to launcher cleanup.

### 17. Infinite health check polling (2026-04-28, FIXED)

`curl -s` with no timeout caused SSH+curl iterations to hang indefinitely during
vLLM health checks. Fixed: `curl --max-time 5` + `ssh -o ConnectTimeout=5`.

### 18. DP>1 base_rank miscalculation (2026-04-28, FIXED)

Launcher didn't pass `vllm_tensor_parallel_size` to the training recipe. With
TP=4 vLLM but recipe defaulting to `tp_size=1`:
- TCPStore `world_size = 1 + 3×1 = 4` (should be `1 + 3×4 = 13`)
- `base_rank` for replica 1 = 2 (should be 5)
- vLLM replica 1 computed `replica_idx = (2-1)//4 = 0` instead of 1
- Replica 1 joined `wsync_cross_0` instead of `wsync_cross_1` → 30-min timeout

Fixed: added `vllm_tensor_parallel_size=${VLLM_TP}` to launcher config overrides.

### 19. WSYNC_CROSS_METHOD hardcoded (2026-04-28, FIXED)

Launcher had `export WSYNC_CROSS_METHOD=gloo` instead of
`export WSYNC_CROSS_METHOD="${WSYNC_CROSS_METHOD:-gloo}"`. The env var from the
PBS script was silently overridden, making XCCL cross-PG impossible to enable.
v16 test ran with gloo cross despite `WSYNC_CROSS_METHOD=xccl` being set.

---

## Key Files

| File | Role |
|------|------|
| `torchtune/dev/vllm_weight_sync_worker.py` | vLLM worker extension: `load_weights_from_path`, `load_weights_from_raw`, `load_weights_from_shm`, `_load_fused_moe_experts`, `init_xccl_communicator`, `receive_weights_xccl_streaming` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Recipe: `_sync_weights_to_vllm()`, `_sync_weights_to_vllm_shm()`, `_init_xccl_weight_sync()`, `_sync_weights_to_vllm_xccl()`, `_wait_for_sync_complete()`, `cleanup()` |
| `torchtune/models/qwen3_moe/_convert_weights.py` | `fuse_experts_for_vllm()` — fuses per-expert gate+up→w13, down→w2 (used by XCCL path) |
| `recipes/configs/dev/production/qwen3_30b_a3b_grpo_xpu.yaml` | Qwen3-30B-A3B MoE GRPO config |
| `recipes/dev/run_grpo_vllm_xpu.sh` | Launcher: sets `VLLM_SERVER_DEV_MODE=1`, `--worker-extension-cls`, `VLLM_GEMMA4` path |
| `recipes/configs/dev/production/qwen3B_gene_recall_xpu.yaml` | `vllm_weight_sync_method: shm` |
| `recipes/configs/dev/production/gemma4_31B_gene_recall_xpu.yaml` | `vllm_weight_sync_method: shm` |
| `recipes/dev/usm_arena_alloc.cpp` | Custom XPU allocator (exact-aligned caching, no arena) |
| `experiments/multinode_32b/run_32b_3node_24way.sh` | 3-node 24-way FSDP launcher (1 vLLM + 2 training) |
| `docs/bugs/intel_ccl_expandable_segments_bug.md` | CCL IPC bug reports (expandable_segments + XPUPluggableAllocator) |

---

## Current Status (2026-04-28)

### 3B (Qwen2.5-3B / Qwen3-3B)

| Aspect | Status |
|--------|--------|
| Best sync method | `xccl` — 3.1s total, fully hidden, 6.2s/step |
| SHM fallback | Validated — 2.8s total, 1.4s waited, 21s/step |
| Production config | `vllm_weight_sync_method: shm` (XCCL not yet default) |
| Gene recall training | 130 steps clean (job 8449766), peak 43.75% success |
| Allocator | `usm_caching_alloc.so` (pluggable, prevents late-step banned:1) |

### 30B MoE (Qwen3-30B-A3B)

| Aspect | Status |
|--------|--------|
| Best sync method | `shm` + GPU transpose + GPU fuse — 3.3s gather, 13s vLLM reload |
| Step time | **35.3s** steady-state (10+2 tiles, batch=1, grpo_samples=4, max_gen=64) |
| Production config | `vllm_weight_sync_method: shm` |
| XCCL (single-node) | **BLOCKED** — UR:40 (IPC handle accumulation on 10 FSDP ranks) |
| Memory | FLAT at 30.41 GiB between steps |

**MoE-specific optimizations** (see `docs/features/moe_integration.md` for details):
- **Fused experts**: 18,867 per-expert params → 531 fused w13/w2 params (training-side fuse)
- **GPU transpose**: vLLM-side `.to(device).transpose(1,2).contiguous()` — 62s → 13s reload
- **Inline GPU fuse**: Fuse gate+up on GPU during gather loop — 18s → 3.3s gather

### 32B (Qwen3-32B)

| Aspect | Status |
|--------|--------|
| Best sync method | 2-hop gloo cross-PG + XCCL intra (dedicated vLLM node) |
| 2-node gloo cross-PG | 47s sync at 1.3 GB/s, ~67s/step. **20/20 clean, exit=0**. CXI leak eliminated |
| 2-node XCCL cross-PG | 9.1s sync at 7.9 GB/s, ~43s/step. **24/24 clean** but leaks ~9 MiB/step |
| **3-node 24-way, gloo×+XCCL_i** | **53.5s avg**. **20/20 clean, 10/10 syncs** (v14). Definitive stability proof |
| 3-node 24-way, XCCL×+XCCL_i | **53.9s avg**. **5/5 clean** (v16b). Sync-step gen 12s vs 20s |
| 3-node 24-way, gloo intra | **72.1s/step avg**. **5/5 clean** (v13). Fallback only |
| 3-node 24-way, **DP=3 parallel** | **59.9s avg**. **5/5 clean** (v18). Benefits at high gen load |
| 3-node 24-way, **pinned buf + G=32** | **~53s avg, 0.60 samples/s**. **Best throughput config** (Test B, 5/5 clean) |
| 3-node 24-way, pinned buf, G=32/max_gen=192 | **~72s avg**. **3/3 clean** (Test G2). Marginal (1.50 GiB free) |
| 3-node 24-way, pinned buf, G=16/max_gen=512 | **~82s**. **3/3 clean** (Test D). At absolute limit (0.01 GiB free) |
| 3-node 24-way, G=48+ or G=64 | **BLOCKED** — 2-chunk rule (G/fbs>2 → CCL explosion); XPU bug at batch=64 |
| Single-node (10+2) | **BLOCKED** — IPC handle accumulation at step 2 |
| Production config | 3-node dedicated vLLM, `vllm_weight_sync_method: xccl`, `WSYNC_INTRA_METHOD=xccl`, **G=32 fbs=16 max_gen=128** |
| DP>1 per-replica PGs | **Validated** (v17/v18, 2026-04-28). Parallel broadcast implemented |

### Summary of weight sync evolution

```
path (safetensors)     →  raw_bytes (/dev/shm)  →  shm (POSIX shared mem)  →  xccl (GPU broadcast)
   ~94s (31B NVMe)          ~20s (31B /dev/shm)       ~19s (31B)                ~8s (31B est.)
   ~9s (3B)                 ~9s (3B)                   ~3s (3B)                  ~3s (3B)

MoE extension (Qwen3-30B-A3B):
per-expert (84s reload)  →  pre-fused (62s)  →  GPU transpose (13s)  →  + GPU fuse (3.3s gather)
   18,867 params             531 params          CPU→GPU transpose       inline fuse during gather
```

Each generation eliminated a bottleneck:
- `path` → `raw_bytes`: removed safetensors serialization overhead
- `raw_bytes` → `shm`: removed file read (zero-copy via shared pages)
- `shm` → `xccl`: removed CPU staging entirely (GPU→GPU broadcast)
- `xccl` → `2-hop`: cross-node gloo + intra-node XCCL; per-replica PGs for DP>1
- 2-hop + deferred broadcast: hide sync latency behind generation (interval=2)
- MoE fused experts: removed per-expert overhead + IPEX shape mismatch bug
- MoE GPU transpose: moved 27 GiB transpose from CPU (20 GB/s) to GPU (1.6 TB/s)
- MoE GPU fuse: moved 48-layer `torch.cat` from CPU (14s) to GPU (<1s)

---

## Potential Next Steps

### Validate DP>1 per-replica PGs

Per-replica intra-PG and cross-PG fix is implemented but untested with DP>1.
Needed for 3-node with multiple vLLM replicas (e.g., TP=4 DP=3 on 12 tiles).
With N replicas, gloo broadcast is N× sequential — may need concurrent threads
or increased `vllm_weight_sync_interval`.

### Default XCCL for 3B production

Switch `qwen3B_gene_recall_xpu.yaml` from `shm` to `xccl`. The XCCL path is
validated and faster (no CPU staging), but SHM is kept as default until XCCL
has more production mileage.

### Long-run 32B stability

2-node gloo cross-PG: 20/20 clean but FSDP collectives leak CXI MR entries
(root cause identified 2026-04-28). Step-28 crash is from FSDP AllGather/ReduceScatter,
not weight sync. Must prevent caching allocator contraction or wait for Intel
SHS 13.1.0. Current mitigation: checkpoint-restart at ~step 60-70.
