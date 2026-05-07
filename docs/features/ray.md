# Ray on Aurora — Setup and Validation

Ray 2.53.0 + vLLM 0.15.0 running on Intel XPU (Aurora). Validated for
single-node TP=4/8 and cross-node TP=16. This document covers the
environment configuration, the Aurora-specific fixes required to make it work,
the smoke test infrastructure, and throughput benchmark results.

The primary use case is `distributed_executor_backend="ray"` in vLLM, which
enables tensor parallel and pipeline parallel placement across more tiles than a
single node provides (12 tiles/node on Aurora, so TP > 12 requires multi-node Ray).

---

## Architecture

```
PBS allocation (N nodes)
├── Head node
│   ├── ray start --head --num-gpus=12 --num-cpus=4
│   └── python3 smoke_ray.py  (driver: ray.init → LLM(distributed_executor_backend="ray"))
└── Worker nodes (N-1)
    └── ray start --address=HEAD:6379 --num-gpus=12 --num-cpus=4  (via SSH)
```

vLLM with `distributed_executor_backend="ray"` places TP workers as Ray actors.
Ray's `IntelGPUAcceleratorManager` assigns `ONEAPI_DEVICE_SELECTOR=level_zero:N`
per actor, isolating each actor to its assigned tile.

For TP=16 across 2 nodes, Ray packs 12 workers on the head node and 4 on the
worker node (greedy placement). The 16-rank AllReduce crosses Slingshot HSN.

---

## Infrastructure

### Smoke tests — `experiments/ray_smoke/`

| File | Purpose |
|------|---------|
| `setup_ray_env.sh` | Sourced on every node — module load, CCL env, resource limits |
| `run_smoke_1node.sh` | 1-node smoke: Ray head + vLLM TP=4 |
| `run_smoke_2node.sh` | 2-node smoke: Ray head + SSH worker + vLLM TP=16 |
| `smoke_ray.py` | Stage 1=cluster health check, Stage 2=vLLM generate |
| `hold_node.sh` | PBS: hold 1 node (debug-scaling queue) |
| `hold_2node.sh` | PBS: hold 2 nodes (debug-scaling queue) |
| `hold_2node_debug.sh` | PBS: run 2-node smoke inside the job (debug queue, 2 nodes) |

### Throughput benchmarks — `experiments/ray_bench/`

| File | Purpose |
|------|---------|
| `bench_ray.py` | Driver: reads `BENCH_*` env vars, times `llm.generate()`, reports tok/s |
| `run_bench_1node.sh` | 1-node benchmark runner (starts Ray head, runs bench_ray.py) |
| `run_bench_2node.sh` | 2-node benchmark runner (PBS_NODEFILE, SSH worker, bench_ray.py) |
| `pbs_bench_1node_32b.sh` | PBS job: Qwen3-32B TP=8 1-node throughput |
| `pbs_bench_2node_32b.sh` | PBS job: Qwen3-32B TP=16 2-node throughput |
| `pbs_bench_2node_480b.sh` | PBS job: Qwen3-Coder-480B TP=16 PP=1 2-node throughput (see 480B section) |

`BENCH_*` variables: `BENCH_MODEL`, `BENCH_TP`, `BENCH_PP`, `BENCH_NUM_PROMPTS`,
`BENCH_INPUT_LEN`, `BENCH_OUTPUT_LEN`, `BENCH_GPU_MEM_UTIL`, `BENCH_MAX_MODEL_LEN`,
`BENCH_WARMUP`. Override in the PBS wrapper before calling `run_bench_*.sh`.

### Running smoke tests

**1-node (interactive, held node):**
```bash
qsub experiments/ray_smoke/hold_node.sh
# SSH into the held node
nohup bash experiments/ray_smoke/run_smoke_1node.sh \
    > /tmp/ray_smoke_1node.log 2>&1 &
tail -f /tmp/ray_smoke_1node.log
```

**2-node (PBS job, smoke runs inside the job):**
```bash
qsub experiments/ray_smoke/hold_2node_debug.sh
# Logs: experiments/ray_smoke/logs/hold_2n_debug.out
#       experiments/ray_smoke/logs/<timestamp>_2node/run.log
```

The 2-node script uses `PBS_NODEFILE` to discover nodes, resolves their HSN
IPs, and SSH-starts the worker raylet. The PBS job approach (`hold_2node_debug.sh`)
is simpler than the held-node approach because `PBS_NODEFILE` is available
inside the job without manual discovery.

---

## Validated Results

### Smoke tests (Qwen3-4B, `enforce_eager=True`, `distributed_executor_backend="ray"`)

| Test | Date | TP | Nodes | Load | Gen | Total |
|------|------|----|-------|------|-----|-------|
| 1-node | 2026-05-06 | 4 | 1 | 26.8s | 1.48s | 37.4s |
| 2-node | 2026-05-06 | 16 | 2 | 88.6s | 3.08s | 104.2s |

### Throughput benchmarks (`experiments/ray_bench/`)

Qwen3-32B (`enforce_eager=True`, `gpu_memory_utilization=0.85`, 64 prompts, 256 in / 128 out tokens):

| Config | TP | PP | Nodes | Tiles | Input tok/s | Output tok/s | Total tok/s |
|--------|----|----|-------|-------|------------|-------------|------------|
| 1-node | 8  | 1  | 1     | 8     | 3,306      | 826.7       | 4,133      |
| 2-node | 16 | 1  | 2     | 16    | 954        | 238.9       | 1,193      |

**TP=16 is 3.5× slower than TP=8** for Qwen3-32B. Cross-node AllReduce on Slingshot
dominates when the model fits on a single node. Use TP=8 (1-node) for Qwen3-32B throughput.
Use TP=16 only when TP=8 is unavailable or for latency-critical deployments with memory pressure.

### Large model loading — Qwen3-Coder-480B-A35B (2-node, 2026-05-07)

Qwen3-Coder-480B-A35B-Instruct: `Qwen3MoeForCausalLM`, 960 GB BF16, 241 safetensor shards,
`vocab_size=151936`, `num_kv_heads=8`. Three distinct blockers encountered across three attempts.

#### TP constraints for this model

Two constraints apply when choosing TP:
- **`vocab_size % TP == 0`** — `VocabParallelEmbedding` requires exact divisibility.
  `151936 = 2^7 × 1187`, so **only power-of-2 TP values are valid**. TP=24, TP=12, TP=20
  all fail with `AssertionError: 151936 is not divisible by <TP>`.
- **`TP % num_kv_heads == 0`** — for TP > num_kv_heads=8, vLLM replicates KV heads
  (instead of partitioning). TP=16 satisfies `16 % 8 == 0`; TP=32 satisfies `32 % 8 == 0`.

#### Lustre page-cache prewarm (`BENCH_PREWARM=1`)

Pre-reads all 241 shards with `find -L | xargs -P 16 dd if=... of=/dev/null bs=4M` on each
node. Populates Linux page cache (1.1 TiB DRAM/node). Total I/O per node: ~448 GiB.
- **Cold** (first read from Lustre): 4.5–5.7 GiB/s per node, both nodes concurrent (~180–200s)
- `stat -L` required to dereference HF blob symlinks; `nohup` + DONE sentinel to survive SSH
  session drops on long transfers.
- With warm cache, safetensors byte-range reads serve from RAM. But: loading is largely
  **compute-bound** (tensor deserialization + XPU DMA transfer), so prewarm helps I/O
  latency, not total loading time.

#### Attempt 1 — TP=8 PP=3 (FAIL: XPU PP KV-cache init bug)

24 workers, weights load successfully (647s worker, 1001s head, warm cache). KV cache
initialization crashes on all 24 workers after weight loading:
```
KeyError: 'model.layers.0.self_attn.attn'
KeyError: 'model.layers.21.self_attn.attn'
KeyError: 'model.layers.42.self_attn.attn'
```
XPU `gpu_model_runner.py` has diverged from the CUDA path; workers receive
`kv_cache_group_spec.layer_names` containing layers from other PP stages, then
`get_layers_from_vllm_config` does an unsafe `forward_context[layer_name]` lookup
→ `KeyError`. See `docs/bugs/vllm_xpu_pp_kvcache_init.md` for full root cause and
monkey-patch fix.

#### Attempt 2 — TP=24 PP=1 (FAIL: vocab size constraint)

All 24 workers crash at model init (before weight loading):
```
AssertionError: 151936 is not divisible by 24
```
`VocabParallelEmbedding._get_indices` in `vocab_parallel_embedding.py:329`. TP=24 contains
a factor of 3; `151936 = 2^7 × 1187` has no factor of 3. TP=24 is architecturally invalid
for this model regardless of hardware.

#### Attempt 3 — TP=16 PP=1 (FAIL: weight loading exceeds walltime)

TP=16 is architecturally valid: `151936 % 16 == 0` ✓, `16 % 8 == 0` ✓ (KV heads replicated
2×). With `gpu_memory_utilization=0.98`: 64 GB × 0.98 = 62.7 GB available; 895/16 = 55.9 GB
weights; ~6.8 GB/tile KV headroom.

Weight loading started but PBS walltime (1 hour) was exceeded. Loaded 96/241 shards (40%)
in ~54 min of loading time. **Extrapolated total: ~135 min (2h15m)** for full 241 shards.

Loading rate is not constant — it degrades as HBM fills:

| Shards loaded | Approx. seconds/shard |
|---------------|----------------------|
| 1–30          | 10–30s (avg ~20s) |
| 30–60         | 25–35s (avg ~30s) |
| 60–96         | 38–52s (avg ~45s) |

Root cause: 16 tiles on 2 nodes all compete for the same host-PCIe bandwidth simultaneously
as HBM fills. Per-shard time grows approximately linearly with the fraction of HBM filled.
A full load would take ~2.5 hours regardless of Lustre prewarm.

#### Summary table

| Config | Failure | Stage |
|--------|---------|-------|
| TP=8 PP=3 | XPU PP KV-cache `KeyError` | KV cache init (post-load) |
| TP=24 PP=1 | `151936 % 24 ≠ 0` | Model init (pre-load) |
| TP=16 PP=1 | Walltime at 40% loaded | Weight loading (~2h15m) |

#### Path forward

**480B cannot be benchmarked in the 1-hour debug queue.** Options:
1. **Production queue** — allocate 3+ hours; TP=16 PP=1 with `gpu_memory_utilization=0.98`
   and `max_model_len=1024` is the configuration to use. `pbs_bench_2node_480b.sh` is
   already configured for this.
2. **Wait for XPU PP fix** — if vLLM upstream fixes `gpu_model_runner.py`'s PP handling,
   TP=8 PP=3 becomes viable (24 tiles, faster per-tile load since each tile loads only 1/3
   of layers). See `docs/bugs/vllm_xpu_pp_kvcache_init.md` for the fix location.
3. **Quantized checkpoint** — INT8/AWQ halves to ~448 GB; TP=8 PP=1 would need 56 GB/tile,
   fitting on one node with no cross-node overhead.

---

## Aurora-Specific Fixes

Five issues had to be resolved to get Ray + vLLM working on XPU. All are
handled by `setup_ray_env.sh` and the launcher scripts — no manual steps needed
for a fresh run.

### 1. `--num-cpus=4` — prevents gRPC worker storm

**Symptom**: `ray.init()` prints "Connected to Ray cluster." then hangs forever.
`/proc/<PID>/wchan` shows `unix_stream_read_generic` on the raylet Unix socket.

**Root cause**: When a driver connects, the raylet pre-starts N Python workers
(N = `num_cpus`, default 96). All 96 workers simultaneously connect to the
dashboard agent gRPC event-aggregator. The gRPC server gets overloaded; workers
time out after 60s; the raylet processes 96+ `GCTaskFailureReason` events
(active indefinitely), blocking its event loop. The driver's
`WorkerRegisterReply` is never sent → driver hangs.

**Fix**: `--num-cpus=4` in `ray start`. Limits pre-started workers to 4, so
the gRPC server isn't flooded. vLLM actors use `num_cpus=0`, so cluster GPU
capacity is unaffected.

### 2. `aiohttp_cors` — required for gRPC event-aggregator to start

**Symptom**: Same hang as above, but earlier in the Ray startup process.
`dashboard_agent.log` shows "Loaded 0 modules." Ray driver log shows
`event_aggregator_client.h:50: Initiating the local event aggregator client`
but the port never opens.

**Root cause**: Without `aiohttp_cors`, Ray's `get_dashboard_dependency_error()`
returns non-None → dashboard agent starts with `--minimal` → gRPC
event-aggregator never starts → CoreWorkerProcess blocks forever.

**Fix**: `setup_ray_env.sh` auto-installs `aiohttp_cors` and `opentelemetry-*`
if missing:
```bash
python3 -c "import aiohttp_cors" 2>/dev/null \
    || pip install --quiet --user aiohttp_cors opentelemetry-api opentelemetry-sdk
```

### 3. `TORCHDYNAMO_DISABLE=1` — prevents Triton crash in Ray actors

**Symptom**: `Unhandled exception: sycl::_V1::exception. what(): No device of
requested type available` from `spirv_utils.so::init_devices`, during
`determine_available_memory → _dummy_run`.

**Root cause**: `VocabParallelEmbedding.forward_xpu` has `@torch.compile`.
Dynamo triggers Triton's `init_devices()` which calls
`sycl::detail::select_device()`. This fails because Ray sets
`ONEAPI_DEVICE_SELECTOR=level_zero:N` per actor — no `opencl:gpu` backend
that Triton needs.

**Fix**: `export TORCHDYNAMO_DISABLE=1`. This is set in `setup_ray_env.sh`
(not just the launcher scripts) so that SSH-started worker raylet processes
also inherit it. vLLM already runs in `-O0`/`enforce_eager` mode, so eager
fallback is correct.

### 4. No `CCL_ZE_IPC_EXCHANGE=drmfd`

**Symptom**: `RuntimeError: oneCCL: CCL_ZE_IPC_EXCHANGE: unexpected value:
drmfd, expected values: sockets, pidfd`.

**Root cause**: `drmfd` was removed from newer oneCCL.

**Fix**: Leave `CCL_ZE_IPC_EXCHANGE` unset. Removed from `setup_ray_env.sh`.

### 5. `ONEAPI_DEVICE_SELECTOR` override for Ray

**Symptom**: `ValueError: Attempting to start raylet with 12 GPU, but
ONEAPI_DEVICE_SELECTOR contains ['gpu']`.

**Root cause**: The `frameworks` module sets
`ONEAPI_DEVICE_SELECTOR=opencl:gpu;level_zero:gpu`. Ray's
`IntelGPUAcceleratorManager` parses the `level_zero:` prefix and gets `["gpu"]`
(length 1), not 12.

**Fix**: Enumerate tile IDs explicitly:
```bash
export ONEAPI_DEVICE_SELECTOR="level_zero:0,1,2,3,4,5,6,7,8,9,10,11"
```
This is done in `setup_ray_env.sh`. Ray actors then receive
`ONEAPI_DEVICE_SELECTOR=level_zero:N` for their assigned tile.

---

## CCL Configuration for Ray Workers

Ray manages process spawning; `mpiexec` is not used. The oneCCL transport must
be `ofi` (not `mpi`), with `FI_PROVIDER=cxi` for Slingshot 11.

```bash
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
```

All 16 TP workers in the 2-node run confirmed receiving the correct CCL env
(logged as "repeated 15x across cluster" in vLLM worker output).

Do **not** use `CCL_PROCESS_LAUNCHER=pmix` or `CCL_ATL_TRANSPORT=mpi` here —
those require `mpiexec --pmi=pmix` and will silently break Ray-spawned workers.
See `memory/feedback_vllm_tp1_ccl_env.md`.

---

## Multiprocessing Guard in Driver Scripts

vLLM forces `VLLM_WORKER_MULTIPROC_METHOD=spawn` when XPU is initialized.
Python's `spawn` re-runs the driver script as `__main__` in each worker
process. Any top-level `LLM()` call in the driver will be hit again.

**Fix**: wrap all driver logic in `if __name__ == "__main__":`. Already done
in `smoke_ray.py`.

---

## Debugging Checklist

1. `ray status --address=$RAY_ADDRESS` — are all expected nodes/GPUs visible?
2. `cat $LOG_DIR/ray_head.log` — did Ray head start? Any GCS errors?
3. `cat $LOG_DIR/ray_worker_*.log` — did workers join? (2-node)
4. `VLLM_HOST_IP` — should be HSN address (10.113.x.x range, not mgmt 10.112.x.x)
5. `no_proxy` — must include all cluster HSN IPs and hostnames
6. `ulimit -u` — should be ≥ 65536
7. Isolate Ray from vLLM: `SMOKE_TP=1` with `distributed_executor_backend="mp"`

For the gRPC hang specifically: check
`/tmp/session_*/logs/dashboard_agent.log` for "Loaded 0 modules." (minimal
mode → fix: install `aiohttp_cors`) vs. normal module loading.
