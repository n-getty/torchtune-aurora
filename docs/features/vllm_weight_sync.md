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

**XCCL for 32B has not been validated in production** — see "32B Memory
Constraints" below for why single-node 32B is currently blocked.

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

---

## Key Files

| File | Role |
|------|------|
| `torchtune/dev/vllm_weight_sync_worker.py` | vLLM worker extension: `load_weights_from_path`, `load_weights_from_raw`, `load_weights_from_shm`, `init_xccl_communicator`, `receive_weights_xccl` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Recipe: `_sync_weights_to_vllm()`, `_sync_weights_to_vllm_shm()`, `_init_xccl_weight_sync()`, `_sync_weights_to_vllm_xccl()`, `_wait_for_sync_complete()`, `cleanup()` |
| `recipes/dev/run_grpo_vllm_xpu.sh` | Launcher: sets `VLLM_SERVER_DEV_MODE=1`, `--worker-extension-cls`, `VLLM_GEMMA4` path |
| `recipes/configs/dev/production/qwen3B_gene_recall_xpu.yaml` | `vllm_weight_sync_method: shm` |
| `recipes/configs/dev/production/gemma4_31B_gene_recall_xpu.yaml` | `vllm_weight_sync_method: shm` |
| `recipes/dev/usm_arena_alloc.cpp` | Custom XPU allocator (exact-aligned caching, no arena) |
| `docs/bugs/intel_ccl_expandable_segments_bug.md` | CCL IPC bug reports (expandable_segments + XPUPluggableAllocator) |

---

## Current Status (2026-04-22)

### 3B (Qwen2.5-3B / Qwen3-3B)

| Aspect | Status |
|--------|--------|
| Best sync method | `xccl` — 3.1s total, fully hidden, 6.2s/step |
| SHM fallback | Validated — 2.8s total, 1.4s waited, 21s/step |
| Production config | `vllm_weight_sync_method: shm` (XCCL not yet default) |
| Gene recall training | 44 steps completed, F1 0→0.405, learning confirmed |
| Allocator | Default (no custom allocator needed — no OOM at 3B) |

### 32B (Qwen3-32B)

| Aspect | Status |
|--------|--------|
| Best sync method (single-node) | `shm` — 19s total, 13s waited |
| XCCL (estimated) | ~8s total, ~0s waited — **not validated** at 32B |
| Single-node blocker | OOM at step 4 (default allocator) or GPU segfault (custom allocator) |
| Colocate-sleep | 100s/step with v2 weight sync — functional but slow |
| Multi-node (2-node HSDP) | 66–81s/step, OFI transport, SHM per-node — **production viable** |
| Best single-node config | G=8/fbs=8, 10+2 tiles, 22.8s/step (3–4 steps before OOM) |

### Summary of weight sync evolution

```
path (safetensors)     →  raw_bytes (/dev/shm)  →  shm (POSIX shared mem)  →  xccl (GPU broadcast)
   ~94s (31B NVMe)          ~20s (31B /dev/shm)       ~19s (31B)                ~8s (31B est.)
   ~9s (3B)                 ~9s (3B)                   ~3s (3B)                  ~3s (3B)
```

Each generation eliminated a bottleneck:
- `path` → `raw_bytes`: removed safetensors serialization overhead
- `raw_bytes` → `shm`: removed file read (zero-copy via shared pages)
- `shm` → `xccl`: removed CPU staging entirely (GPU→GPU broadcast)

---

## Potential Next Steps

### Validate XCCL at 32B scale

XCCL broadcast was validated for 3B (5.75 GiB, 434 params, 3.1s). For 32B
(59 GiB, ~700 params), the expected time is ~8s at 2.2 GB/s cross-process
bandwidth. This would eliminate the 13s waited time that SHM incurs. However,
32B single-node training is currently blocked by the OOM/allocator issue — XCCL
weight sync can only be validated once training itself is stable.

### Dedicated vLLM node for 32B

Use 1 node (12 tiles) for FSDP training with default allocator, 1 node for
vLLM. This eliminates vLLM memory contention from the training node, potentially
avoiding the step-4 OOM. Weight sync would use SHM or raw_bytes to a cross-node
file path (Lustre or /dev/shm with RDMA). Not yet tested.

### Reduce cross-process XCCL bandwidth gap

Isolated XCCL broadcast achieves 14 GB/s but cross-process only 2.2 GB/s
(6.4× gap). Investigating whether this is a Level Zero IPC overhead, TCPStore
rendezvous cost, or XCCL internal serialization.

### Default XCCL for 3B production

Switch `qwen3B_gene_recall_xpu.yaml` from `shm` to `xccl`. The XCCL path is
validated and faster (no CPU staging), but SHM is kept as default until XCCL
has more production mileage.
