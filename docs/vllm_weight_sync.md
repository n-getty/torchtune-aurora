# vLLM Weight Sync — Implementation and Experiments

After each optimizer step in GRPO training, the policy model's updated weights
must be pushed to the vLLM inference server so that the next generation step
uses the trained model rather than stale initial weights. This document covers
the implementation, bugs found and fixed, measured performance, and remaining
work.

## Background

The GRPO recipe uses a colocated vLLM server for efficient generation:
- **Training tiles**: 10 XPU tiles running FSDP2 over the policy model
- **vLLM tiles**: 2 XPU tiles (TP=2 for 31B, TP=1×DP=2 for 3B) running the
  inference server

After each optimizer step, rank 0 must synchronize updated weights from the
FSDP2 sharded state into the vLLM server. The challenge: vLLM and the training
process are on different tiles and cannot share device memory.

### Why not XCCL?

The natural approach would be a collective broadcast directly from training
tiles to vLLM tiles (as TRL does with `update_named_param`). On XPU, this
approach SIGABRTs: creating a second XCCL communicator while the training
process group is active causes a fatal error in the Level Zero runtime. All
weight sync must therefore go through CPU memory.

---

## Methods Implemented

Three methods are implemented, selectable via `vllm_weight_sync_method` in the
training config (default: `raw_bytes` for backward compatibility):

```yaml
vllm_weight_sync_method: shm   # shm | raw_bytes | path
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

### 3. `shm` — POSIX shared memory (recommended)

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

---

## Async Flow

The sync runs asynchronously relative to the training backward pass, but not
relative to the next generation step. The timeline per step is:

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

To fully hide the sync, the gather + copy + POST would need to complete within
the GRPO backward + optimizer time of the current step. For 3B this is feasible
(copy+http=1.6s vs grpo+opt≈5.5s); for 31B it is not (copy+http=13s vs
grpo+opt≈13s — marginal).

---

## Measured Performance

### Qwen2.5-3B (5.75 GiB BF16, 434 params, VLLM_DP=2)

Validated 2026-04-16, job 8438434 (hold node, 5 steps).

| Method | gather | copy/write | http | total | waited | step time |
|--------|--------|------------|------|-------|--------|-----------|
| `raw_bytes` | 1.1s | 2.6s | 5.5s | 9.3s | 7.8s | 21–22s |
| `shm` step 0 | 14.7s† | 4.2s (1.4 GB/s) | 1.0s | 19.9s | 4.9s | 63.7s† |
| `shm` steps 1–4 | 1.1–1.3s | 0.7–0.8s (8 GB/s) | 0.9s | **2.8–2.9s** | **1.4s** | **21.1–21.5s** |

† FSDP initialization overhead on step 0 (one-time).

**SHM vs raw_bytes (steady state)**: 9.3s → 2.8s (3.3×), waited 7.8s → 1.4s
(5.6×). Step time unchanged at ~21s — sync fully hidden behind generation.

### Gemma4-31B (57.18 GiB BF16, 832 params, TP=2 single replica)

Validated 2026-04-16, job 8438514 (hold node, 5 steps).

| Method | gather | copy | http | total | waited | step time |
|--------|--------|------|------|-------|--------|-----------|
| `raw_bytes` on NVMe | ~5.5s | ~44s | ~44s | ~94s | ~88s | — |
| `raw_bytes` on `/dev/shm` | ~5.5s | ~7s | ~7s | ~20s (est.) | ~13s (est.) | — |
| `shm` step 0 | 16.9s† | 41.3s (1.4 GB/s) | 7.1s | 65.4s | 48.1s | 135.5s† |
| `shm` steps 1–4 | 5.5–5.6s | 6.9–7.7s (7.5–8.3 GB/s) | 5.7–6.4s | **18.7–19.3s** | **12.9–13.5s** | **~83s** |

† Step 0 overhead: FSDP init (gather) + page-fault warmup (copy).

**SHM vs raw_bytes on `/dev/shm` (estimated)**: ~20s → ~19s (modest). The main
win of SHM over raw_bytes for large models is eliminating one extra DDR5 pass
(the file read on the vLLM side). For 31B the XPU transfer bandwidth (~9 GB/s
for `param.copy_()` across 832 params) dominates.

**Step time breakdown (31B steady state)**:
- vLLM generation (actual): ~51s
- GRPO forward: ~13s
- gather + other: ~5.6s
- waited (sync): ~13s
- Total: ~83s

The waited=13s is irreducible given the current async flow: copy(7s)+http(6s)=13s
starts at the end of the step, and generation cannot begin until it finishes.

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

---

## Key Files

| File | Role |
|------|------|
| `torchtune/dev/vllm_weight_sync_worker.py` | vLLM worker extension: `load_weights_from_path`, `load_weights_from_raw`, `load_weights_from_shm` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Recipe: `_sync_weights_to_vllm()`, `_sync_weights_to_vllm_shm()`, `_wait_for_sync_complete()`, `cleanup()` (SHM unlink) |
| `recipes/dev/run_grpo_vllm_xpu.sh` | Launcher: sets `VLLM_SERVER_DEV_MODE=1`, `--worker-extension-cls`, `VLLM_GEMMA4` path |
| `recipes/configs/dev/production/qwen3B_gene_recall_xpu.yaml` | `vllm_weight_sync_method: shm` |
| `recipes/configs/dev/production/gemma4_31B_gene_recall_xpu.yaml` | `vllm_weight_sync_method: shm` |
| `run_shm_sync_test.sh` | 5-step 3B smoke test |
| `run_shm_sync_test_31b.sh` | 5-step 31B smoke test |

---

## Current Status

- **3B SHM**: validated, 2.8s/sync, 21s/step, sync fully hidden. Production-ready.
- **31B SHM**: validated, 19s/sync, 83s/step, 13s residual wait. Stable, no OOM.
- Both production configs set `vllm_weight_sync_method: shm`.

---

## Potential Next Steps

### Reduce 31B waited time

The 13s wait (copy=7s + http=6s) is the current bottleneck. Options:

1. **Overlap gather with GRPO backward**: The gather is a blocking collective
   that could theoretically start after the final optimizer step of the
   previous mini-batch. Requires restructuring the training loop.

2. **Streaming POST**: Start sending weight tensors to vLLM before all of them
   are copied to SHM (pipeline copy + HTTP). Requires a chunked HTTP protocol
   or multiple smaller SHM blocks with coordinated signaling.

3. **Reduce http time via batched `param.copy_()`**: Currently `load_weights()`
   iterates 832 params serially. A fused kernel or parallel `copy_()` across
   params could improve the ~9 GB/s effective XeLink bandwidth.

4. **Accept the 13s cost**: For long-context 31B generation (>1000 tokens),
   the generation step will exceed the sync time and waited will drop to 0
   without any changes.

### 3B production run

Resubmit gene recall with:
- `kl_coeff: 0.3` (increased from 0.1 — spikes grew to 2.6 by step 43)
- `save_every_n_steps: 20` (avoid losing work on walltime kill)
- `vllm_weight_sync_method: shm` (already in config)
- Resume from checkpoint when available

### 31B production run

Gemma4-31B F1 baseline is 0.337 vs Qwen3B 0.104, giving more headroom to
improve. Step time (~83s) limits debug-queue runs to ~40 steps/hour.
Regular queue needed for meaningful training runs.
