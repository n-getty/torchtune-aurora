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
| `pbs_serve_bench_2node_480b_tp8pp3.sh` | PBS job: 480B TP=8 PP=3 serve bench — FAILS (sharded_state PP>1 incompatible) |
| `pbs_serve_bench_3node_480b_tp32.sh` | PBS job: 480B TP=32 PP=1 3-node serve bench — validated 106.19 tok/s |

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

**Qwen3-32B** (`enforce_eager=True`, `gpu_memory_utilization=0.85`, 64 prompts, 256 in / 128 out tokens):

| Config | TP | PP | Nodes | Tiles | Input tok/s | Output tok/s | Total tok/s |
|--------|----|----|-------|-------|------------|-------------|------------|
| 1-node | 8  | 1  | 1     | 8     | 3,306      | 826.7       | 4,133      |
| 2-node | 16 | 1  | 2     | 16    | 954        | 238.9       | 1,193      |

**TP=16 is 3.5× slower than TP=8** for Qwen3-32B. Cross-node AllReduce on Slingshot
dominates when the model fits on a single node. Use TP=8 (1-node) for Qwen3-32B throughput.
Use TP=16 only when TP=8 is unavailable or for latency-critical deployments with memory pressure.

**Qwen3-Coder-480B-A35B** (`enforce_eager=True`, `load_format=sharded_state`, `vllm bench serve`,
input=1024, output=512, rate=inf, 2026-05-15):

| Config | Nodes | Tiles | max_num_seqs | Output tok/s | Peak tok/s | Server load | TTFT mean | TPOT |
|--------|-------|-------|-------------|-------------|-----------|-------------|-----------|------|
| TP=16 PP=1 (baseline) | 2 | 16 | 24 | 78.55 | 120.00 | 870s | — | — |
| **TP=32 PP=1** | **3** | **32** | **64** | **106.19 (+35%)** | **192.00** | **320s** | 226s | 515ms/tok |
| TP=8 PP=3 | 2 | 24 | — | FAIL | — | — | — | — |

TP=32 advantages: 2× decode tiles → higher parallel decode throughput; ~34 GiB free HBM/rank
(vs ~4 GiB at TP=16) → 2.7× more concurrent sequences; 2.7× faster model load.
TTFT=226s at rate=inf is prefill-bound (128 concurrent 1024-token prompts). TPOT=515ms/token
is the relevant number for interactive workloads. **TP=32 PP=1 is the production config.**

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

#### Attempt 1 — TP=8 PP=3 (FAIL: two independent blockers)

This config hits two distinct bugs, each independently fatal. See
`docs/bugs/vllm_xpu_pp_kvcache_init.md` for full root cause and fix details.

**Failure B — sharded_state loader (2026-05-15):** When using `load_format=sharded_state`
with PP>1, all PP stage k≠0 workers crash during weight loading:
```
KeyError: 'model.layers.0.input_layernorm.weight'
  File "vllm/model_executor/model_loader/sharded_state_loader.py", line 141
```
Root cause: the sharded_state converter stores all N layers with absolute indices in each
rank file (PP=1 design). Stage 1/2 workers' `model.state_dict()` only contains their
local-stage layers — the first key read from the file (`model.layers.0.*`) is absent.
Not fixable without converter changes to produce per-stage rank files.

**Failure A — KV cache init (2026-05-07, HF format):** Even with HF format (avoiding the
loader issue), 24 workers crash post-load during KV cache initialization:
```
KeyError: 'model.layers.0.self_attn.attn'
KeyError: 'model.layers.21.self_attn.attn'
KeyError: 'model.layers.42.self_attn.attn'
```
XPU `gpu_model_runner.py` has diverged from the CUDA path; workers receive
`kv_cache_group_spec.layer_names` containing layers from other PP stages, then
`get_layers_from_vllm_config` does an unsafe `forward_context[layer_name]` lookup → `KeyError`.

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

Per-shard wall time, computed from rank-0's tqdm timestamps (not its EWMA `s/it` rate):

| Shards    | Wall time            | Pattern |
|-----------|---------------------|---------|
| 1–12      | 39 s total           | warm-up burst |
| 13+       | bimodal: ~5 s OR 30–50 s | most shards stall |

Average of ~40 s/shard from shard 13 onward, **constant** — not growing with HBM fill.
The earlier "20 s → 45 s" reading was the tqdm exponential-moving-average rate
catching up to the stall regime after the initial warm-up burst, not per-shard
wall time growing linearly with HBM occupancy.

#### Root cause: reading from Lustre (not staging to /tmp tmpfs)

A diagnostic experiment (`experiments/load_diag/`) verified each candidate.
The 480B benchmark reads directly from `/flare/datasets/...` (Lustre);
production training launchers (`recipes/dev/run_qwen3_30b_vllm_server.sh:65`)
copy the model to `/tmp/torchtune/...` (Aurora compute `/tmp` is **tmpfs**,
504 GiB RAM-backed, NUMA-interleaved) before starting vLLM. The difference
is dramatic.

**Direct-Lustre vs `/tmp` tmpfs vs Copper (multiple models, vLLM TP=8 single-node):**

| Model | Source | `Loading weights took` | Speedup | Notes |
|-------|--------|-----------------------:|--------:|-------|
| Qwen3-30B-A3B (60 GB) | Lustre, cold | **785.83 s** (Exp 4) | 1× | |
| Qwen3-30B-A3B | `/tmp` tmpfs | **13.13 s** (Exp 5) | 60× | page cache warm from Exp 4 |
| Qwen3-30B-A3B | production baseline (fresh node, TP=2) | **9.13 s** | — | |
| Llama-3.3-70B (132 GB) | Lustre, cold | **221.66 s** (Exp 6) | 1× | |
| Llama-3.3-70B | `/tmp` tmpfs (after `cp -rL`, 85 s stage) | **32.68 s** (Exp 6) | **6.8×** | |
| Llama-3.3-70B | Copper FUSE, page cache warm | **17.33 s** (Exp 6) | **12.8×** | **artifact** — see below |
| Llama-3.3-70B | Copper FUSE, cold node | **212.73 s** (Exp 9b) | **1.0×** | fresh node; matches Lustre |
| Llama-3.3-70B | Copper FUSE, same-node 2nd load | **13.14 s** (Exp 10) | **16.9×** | page cache warm from 1st load |
| Llama-3.3-70B | Copper FUSE, cross-node 2nd load | **385.22 s** (Exp 11) | **0.6×** | **1.74× slower** than Lustre direct |
| Llama-3.3-70B | sharded_state from /tmp tmpfs (Exp 13) | **13.23 s** | **16.7×** | each rank reads only its 16 GB slice |
| Qwen3-30B-A3B (MoE, 60 GB) | sharded_state from /tmp tmpfs (Exp 17) | **5.52 s** | **142×** | CPU converter; gate key + XPU scale buffer fixes required |

#### Copper FUSE — evaluated and rejected for cross-node loading

Exp 6's 17.33 s Copper result was a **warm-cache artifact**: the script ran
Lustre-direct first (221 s, warming the Linux page cache), then tmpfs (32 s),
then Copper (17 s) on the same node. Copper read from already-warm page cache.

Exp 9b confirmed this: on a fresh node with cold page cache, Copper loaded
70B in **212.73 s** — identical to Lustre direct.

Exp 10 (same-node cold→hot): cold load = **69.46 s** (partially warm cache),
hot load = **13.14 s** (both Copper cache and Linux page cache warm). The
hot-load speedup is indistinguishable from Linux page cache warming.

Exp 11 (cross-node cache test — the critical measurement for 405B): after
loading on HEAD node (populating Copper cache), loading on WORKER node
(cold page cache) took **385.22 s** total — **1.74× slower than Lustre
direct** (221 s). Per-shard timing showed 8–10 s for early shards
then a **148 s stall on a single shard** at 73%, consistent with the
bimodal Lustre I/O stall pattern amplified by FUSE + Thallium RPC overhead.

**Copper is NOT viable for cross-node model loading on Aurora.** The
cooperative cache does not serve cross-node reads at useful bandwidth.
The only scenario where Copper helps is same-node re-loads, which Linux
page cache already provides for free.

**One viable approach: `/tmp` tmpfs staging.**

- **`/tmp` tmpfs staging** — what production already does. Reliable. 6–60× speedup.
  Limited to models that fit in 504 GiB tmpfs per node. Use `cp -rL` (not `cp -r`)
  to dereference HF blob symlinks; otherwise the copy creates broken symlinks
  and vLLM falls back to Lustre or fails.

**What's NOT the cause** (each pre-tested before Exp 5 found the real issue):

| Hypothesis                              | Verdict      | Evidence |
|-----------------------------------------|--------------|----------|
| Shard non-uniformity                    | ❌ ruled out | All 241 shards = 3.99 GB ± noise |
| Lustre I/O bandwidth                    | ❌ ruled out | Exp 1: 16 procs × 60 shards, per-shard time stable ~1.5 s |
| L0/USM allocator fragmentation          | ❌ ruled out | Exp 2: 256 MiB alloc probe constant 0.2 ms across HBM 4.5 → 48 GB |
| H2D / PCIe degradation as HBM fills     | ❌ ruled out | Exp 2: per-shard H2D **decreased** 8.15 → 2.08 s as HBM filled |

The original "PCIe contention as HBM fills" reading was wrong: hardware H2D
sustains ~2 GB/s/tile, allocator latency stays at 0.2 ms, and Lustre I/O
itself measures fine when read independently. The misdiagnosis came from
trusting tqdm's EWMA `s/it` rate as if it were per-shard wall time.

#### Why tmpfs is faster than Lustre page cache

This is the part that's not yet fully explained. Both end up in DRAM
(Lustre's reads land in the Linux page cache; tmpfs IS DRAM), so naively
they should perform equivalently. They don't. The 60× gap remains even
after `BENCH_PREWARM=1` populates the page cache. Working hypotheses
(verifiable with `numastat`, `perf stat -e dTLB-load-misses`):

- **NUMA placement**: tmpfs is mounted with `mpol=interleave:0-1`, so pages
  spread evenly across the two sockets. Lustre page cache lands on whichever
  socket the reading thread ran on, then DMAs cross-socket.
- **mmap fault overhead**: Lustre-backed mmap pages may take more expensive
  faults (extent-list lookups, OST coordination) than anonymous tmpfs pages,
  even after the cache hit.
- **Hugepages**: tmpfs supports transparent hugepages straightforwardly;
  Lustre-backed mmap may not. Per-page TLB pressure on 940 GB matters.

#### Summary table

| Config | Nodes | Outcome | Stage | Output tok/s |
|--------|-------|---------|-------|-------------|
| TP=8 PP=3 | 2 | FAIL (Failure A + B) | Weight load / KV cache init | — |
| TP=24 PP=1 | 2 | FAIL (`151936 % 24 ≠ 0`) | Model init (pre-load) | — |
| TP=16 PP=1 | 2 | PASS (baseline) | — | 78.55 |
| **TP=32 PP=1** | **3** | **PASS (recommended)** | — | **106.19 (+35%)** |

#### Path forward — for 480B specifically

**RESOLVED (2026-05-15)**: TP=32 PP=1 with sharded_state format is the validated
production config (106.19 output tok/s, +35% vs TP=16 baseline). See throughput
benchmark results below.

Production training launchers stage to `/tmp` (tmpfs) and load in seconds.
The 480B benchmark cannot do the same trivially because **480B = 940 GB > 504 GiB
tmpfs per node**. Options (historical context for how we got here):

The two approaches combine naturally: use `save_sharded_state` once (options 1/4),
then distributed tmpfs staging on every subsequent run (option 2).

**1. One-time save: CPU-side offline converter (exp16) — preferred**

`experiments/load_diag/convert_hf_to_sharded.py` reads the HF safetensors
directly, applies FusedMoE packing + TP sharding in CPU RAM, and writes
per-rank `model-rank-{rank}-part-0.safetensors` files. No GPU, no vLLM
engine start, no FusedMoE Python overhead.

```bash
# Hold a compute node (needs 1134 GiB DRAM for page cache)
qsub experiments/load_diag/hold_1node.sh

# Convert: 4 parallel processes, each handling 4 ranks sequentially
bash experiments/load_diag/exp16_convert_480b.sh <JOBID>
# Runtime: ~12-15 min (10 min cold Lustre read, then page-cached passes)
# Output: 16 × ~59 GB rank files on Lustre
```

Why this beats the vLLM-based save (exp15):

| Method | One-time cost | Root cause |
|--------|--------------|------------|
| exp15 vLLM engine | ~3-4h | FusedMoE.weight_loader: 42s/shard Python overhead (21× vs 2s hardware ceiling) |
| exp16 CPU converter | ~12-15 min | Direct slice + pack in numpy, no vLLM overhead |

Shape transformations applied by the converter:
- `experts.{j}.gate/up_proj.weight [I, H]` → `w13_weight [E, 2*I_p, H]` (column-parallel)
- `experts.{j}.down_proj.weight [H, I]` → `w2_weight [E, H, I_p]` (row-parallel)
- `q/k/v_proj.weight` → `qkv_proj.weight [Q_p+KV_p+KV_p, H]` (stacked, column-parallel)
- `o_proj.weight [H, Q]` → `o_proj.weight [H, Q_p]` (row-parallel)
- norms, routers, q/k_norm → replicated (copied to each rank)
- `embed_tokens / lm_head` → vocab-parallel (`[vocab//tp, H]` per rank)

**2. Distributed tmpfs staging + fast reload (every subsequent run)**

With per-rank files on Lustre, each 2-node run stages only what it needs:

```bash
# Node 0 stages ranks 0-7  (~470 GB → local /tmp)
# Node 1 stages ranks 8-15 (~470 GB → local /tmp)
# Both nodes stage metadata (config.json, tokenizer, etc.)
```

Then load with `load_format=sharded_state`, TP=16, Ray executor:
```python
LLM(model="/tmp/.../sharded_480b_tp16/",
    load_format="sharded_state",
    tensor_parallel_size=16,
    distributed_executor_backend="ray")
```

Each rank's `glob.glob` runs on **local tmpfs** (fast, no Lustre hang risk).
All 16 ranks load their ~59 GB slices in parallel from local HBM-adjacent DRAM.
Expected load time: **~15–20 s** (vs ~2.5 h Lustre cold, ~32 s tmpfs staged TP=8
at 70B scale). Staging cost: ~200 s parallel cp on both nodes (each copies 470 GB
at ~2 GB/s Lustre read bandwidth).

**Validation status:**
- **70B TP=8 single-node**: **VALIDATED** (Exp 12+13, 2026-05-13)
  - Save: 73s load (tmpfs) + **31.7s write** (8 × 16 GB to Lustre)
  - Reload: **13.23s** `Loading weights took` (vs 221s Lustre cold — **16.7× speedup**)
  - Staging cost (Lustre → /tmp): 97s (one-time per hold, not per vLLM restart)
- **30B-A3B MoE TP=8 single-node (CPU converter)**: **VALIDATED** (Exp 17, 2026-05-13)
  - Convert: ~12 min CPU-only on compute node (`exp16_convert_480b.sh` pattern)
  - Reload: **5.52s** `Loading weights took` (vs 785s Lustre cold — **142× speedup**)
  - Staging cost (Lustre → /tmp): 41s
  - MoE-specific fixes in converter: (1) gate key `mlp.gate.weight` →
    `mlp.experts._gate.weight` (`_filter_subtensors` lexicographic tie-breaking);
    (2) 4 XPU attention scale buffers `self_attn.attn._k/_v/_q/_prob_scale`
    (float32 1.0, registered by vLLM `Attention` module, absent from HF checkpoints)
- **480B TP=16 2-node (sharded_state, Lustre)**: **VALIDATED** (2026-05-15)
  - Load via Ray, `load_format=sharded_state`, TP=16, `max_num_seqs=24`
  - Output tok/s: **78.55** (baseline), server load: 870s
- **480B TP=32 3-node (sharded_state, Lustre)**: **VALIDATED** (2026-05-15)
  - Load via Ray, `load_format=sharded_state`, TP=32, `max_num_seqs=64`
  - Output tok/s: **106.19 (+35%)**, server load: 320s; peak output tok/s: 192
  - TTFT=226s mean at rate=inf (prefill-bound, 128 concurrent 1024-token prompts)
  - TPOT=515ms/token; bench logs: `ray_bench/logs/20260515_133332_2node_serve/`

**Scripts:**

| Script | Purpose |
|--------|---------|
| `experiments/load_diag/convert_hf_to_sharded.py` | CPU converter: HF → vLLM sharded_state |
| `experiments/load_diag/exp16_convert_480b.sh` | Driver: parallel converter on held node |
| `experiments/load_diag/exp12_save_sharded.py` | Alt: load model via vLLM, save sharded state |
| `experiments/load_diag/exp13_load_sharded.py` | Load from sharded state |
| `experiments/load_diag/exp12_13_run.sh` | 70B save+load validation (1 node, 1h) |
| `experiments/load_diag/exp14_distributed_stage_load.sh` | 2-node distributed staging + load |

3. **Quantized checkpoint** — INT8/AWQ halves to ~448 GB; fits in one node's
   tmpfs after staging, TP=8 PP=1 single-node eliminates cross-node entirely.

4. **Wait for XPU PP fix** — TP=8 PP=3 (24 tiles) is blocked by two independent
   bugs (sharded_state loader format incompatibility + XPU `gpu_model_runner.py`
   KV cache init divergence). Even if both are fixed, PP>1 with sharded_state requires
   the converter to produce per-stage rank files. See `docs/bugs/vllm_xpu_pp_kvcache_init.md`.
   **TP=32 PP=1 (+35% throughput) is the preferred path.**

**Note:** `BENCH_PREWARM=1` warms the Lustre page cache but did NOT bring
load time anywhere close to staging — production with `/tmp` staging hits
~9 s on Qwen3-30B; cold Lustre with the same vLLM stack took 786 s on the
same model. The page cache and tmpfs are both DRAM-resident; the gap
between them (mmap fault cost, NUMA placement, TLB, hugepage support) is
the next thing to investigate, but the practical fix is "stage to `/tmp`".

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
