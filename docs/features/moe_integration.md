# MoE Integration — Aurora/XPU

MoE model support on Aurora HPC (Intel Max Series GPUs / XPU). Two workstreams:

1. **Qwen3-30B-A3B** — E2E GRPO training with vLLM weight sync (production-ready)
2. **Gemma4 26B-A4B Expert Parallelism** — EP=4/DP=3 on 12 tiles (research, backward blocked)

---

## Qwen3-30B-A3B — E2E GRPO with MoE Weight Sync

### Architecture

Qwen3-30B-A3B: 36 transformer layers, 48 MoE layers (every even layer is MoE).
Each MoE layer: 128 experts (top-8 active), `gate_proj` + `up_proj` + `down_proj`
per expert. Total model size: 56.87 GiB (BF16).

**Training layout**: 10 tiles FSDP2 (XCCL) + 2 vLLM tiles (TP=2), single Aurora node.
SHM weight sync via `/dev/shm/torchtune/weight_update.raw`.

### MoE Weight Sync Challenge

MoE models have thousands of small per-expert weight tensors (18,867 for Qwen3-30B-A3B).
Two problems:

1. **vLLM-side**: IPEX's `_IPEXGatedMLPMOEXPU` transposes w13/w2 expert weights in-place
   during first forward. After that, `weight_loader` cannot handle the post-prepack shapes.
   `model.load_weights()` on 18,867 individual expert tensors triggers this bug.
2. **Training-side**: Gathering 18,867 FSDP shards and copying them individually to SHM is
   slow (high Python dispatch overhead).

### Solution: Fused Expert Weight Sync

Training side fuses per-expert gate+up projections into w13 and keeps down as w2
before sending to vLLM. This reduces 18,867 params to 531 fused params.

**Training-side flow** (`_sync_weights_to_vllm_shm()` in recipe):
1. `full_tensor()` gather — all FSDP ranks participate
2. Inline GPU fuse: as gate/up/down experts arrive for each layer, fuse `torch.cat([gate, up], dim=1)` on GPU, then `.cpu()`
3. Background thread: `ctypes.memmove()` fused tensors into SHM block
4. POST metadata to vLLM `/collective_rpc`

**vLLM-side flow** (`load_weights_from_shm()` in `vllm_weight_sync_worker.py`):
1. Detect fused expert params via regex (`model.layers.N.mlp.experts.w13_weight`)
2. Route fused experts to `_load_fused_moe_experts()` (bypasses `weight_loader`)
3. TP-shard experts: slice by `tp_rank * shard_size : (tp_rank+1) * shard_size`
4. GPU transpose: `.to(device).transpose(1,2).contiguous()` — detects IPEX transpose via shape comparison
5. `param.data.copy_()` in-place

### Optimization History

**vLLM-side reload** (56.87 GiB, 48 MoE layers):

| Version | Method | Time | Bottleneck |
|---------|--------|------|------------|
| v1 | Per-expert tensors (18,867 params) | 82-86s | Stack+TP+transpose per layer |
| v2 | Pre-fused w13/w2 (531 params) | 62s | CPU transpose ~27 GiB at 20 GB/s = 55s |
| v3 | GPU transpose | **13s** | `.to(device)` before `.transpose(1,2).contiguous()`, 1.6 TB/s GPU vs 20 GB/s CPU |

**Training-side gather** (FSDP AllGather + fuse + SHM copy):

| Version | Method | Time | Bottleneck |
|---------|--------|------|------------|
| v1 | CPU fuse (separate `fuse_experts_for_vllm()`) | 18s | `torch.cat` on CPU for 48 layers = 14s |
| v2 | Inline GPU fuse during gather loop | **3.3s** | Fuse on GPU as experts arrive; `.cpu()` per-layer |

**End-to-end step time** (step 2 steady-state, max_gen=64):

| Version | G | vLLM reload | Gather | Step total | "Other" |
|---------|---|-------------|--------|------------|---------|
| Baseline (per-expert) | 4 | 84s | 4s | 95s | 64s |
| Pre-fused | 4 | 62s | 4s | ~80s | ~50s |
| GPU transpose | 4 | 13s | 18s | 50s | 18s |
| GPU fuse | 4 | 13s | 3.3s | 35.3s | 3.3s |
| **G=8 (production)** | **8** | **13s** | **3.3s** | **54.8s** | **3.3s** |

G=8 step time is higher (54.8s vs 35.3s) but processes 2× sequences/step → **1.8× throughput** (9.2 vs ~5 tok/s). G=4 OOMs at step 1; G=8 is the only stable config.

### Key Technical Details

**GPU transpose in `_load_fused_moe_experts()`**: IPEX transposes w13/w2 weights
in-place during `_IPEXGatedMLPMOEXPU.__init__()`. After that first forward, the
parameter shapes are permanently transposed. The reload function detects this via
`w13_param.shape[1] != w13_tp.shape[1]` and transposes on GPU. `.to(device)` is a
no-op if the tensor is already on the correct device (XCCL path).

**Inline GPU fuse**: During the FSDP gather loop, expert tensors stay on GPU. When
all 3 projections (gate, up, down) for a layer arrive, gate+up are fused via
`torch.cat([gate, up], dim=1)` on GPU, then moved to CPU. Peak GPU overhead: ~1.9 GiB
per layer (source + fused result). Non-expert params go straight to `.cpu()`.

**BMM expert forward**: `GroupedExpertsHF` uses scatter-pad-bmm-gather path
(not sequential per-expert loops). GRPO fwd+bwd across 48 MoE layers: ~23s.
See `project_bmm_expert_speedup.md` for the 6.3x speedup details.

### XCCL Weight Sync for MoE

XCCL Mode 0 in the recipe also supports fused experts: the gather-into-dict flow
calls `fuse_experts_for_vllm()` after gathering, then builds the manifest and batches
from fused names/shapes. The vLLM XCCL receive side routes fused experts to
`_load_fused_moe_experts()` via the same regex detection.

**Single-node XCCL: BLOCKED** — `full_tensor()` AllGather on 10 FSDP ranks triggers
UR:40 (IPC handle accumulation exhausts L0 driver cache). Same known issue as 32B dense.
Only viable in 2-node HSDP mode (5 ranks/node).

### Files Modified (MoE weight sync)

| File | Change |
|------|--------|
| `torchtune/dev/vllm_weight_sync_worker.py` | `_load_fused_moe_experts()` with GPU transpose; fused expert routing in `load_weights_from_shm()` and `receive_weights_xccl_streaming()` |
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Inline GPU fuse in SHM path; gather-fuse-batch flow in XCCL Mode 0 |
| `torchtune/models/qwen3_moe/_convert_weights.py` | `fuse_experts_for_vllm()` utility (used by XCCL path) |

### Validation (2026-04-27)

| Test | Config | Status | Notes |
|------|--------|--------|-------|
| SHM + GPU transpose + GPU fuse | 10+2 tiles, G=4, 3 steps | **PASS** | 35.3s/step steady-state; 3.3s gather; 13s vLLM reload |
| Memory stability | G=4, 3 steps | **PASS** | FLAT at 30.41 GiB between steps |
| XCCL single-node | 10+2 tiles, step 0-1 | **BLOCKED** | UR:40 at step 2 (IPC handle accumulation) |

### P0/P1 Config Tuning + torch.compile (2026-04-27)

Four sequential tests on a single debug-scaling node, shared vLLM server (TP=2).

| Test | batch | G | fbs | compile | Steps | Step Time | Throughput | Result |
|------|-------|---|-----|---------|-------|-----------|------------|--------|
| A (baseline) | 1 | 4 | 4 | False | 1/3 | 50.9s | ~5 tok/s | OOM step 1 |
| B (batch=2) | 2 | 4 | 4 | False | 0/3 | — | — | OOM step 0 bwd |
| **C (G=8)** | **1** | **8** | **8** | **False** | **3/3** | **54.8s** | **9.2 tok/s** | **PASS** |
| D (compile) | 1 | 4 | 4 | True | 0/3 | — | — | SYCL compile timeout |

**Winner: G=8** — 2× RL samples per step, 1.8× throughput vs G=4.

**Test C (G=8) details** (steady-state step 2):
- TIMING: total=54.8s, gen=7.6s, grpo=43.5s, clip=0.2s, opt=0.2s, other=3.3s
- GENTIMING (warm): vllm=3.1s, policy_fwd=2.1s, ref_fwd=2.2s
- Weight sync: gather=3.3s, copy=7.1s (8.1 GB/s), http=14.6s → total=25.0s
- Memory: peak_resv=62.43 GiB on rank 6 (0.41 GiB free). Stable after step 1.

**Why G=8 survives but G=4 OOMs at step 1**: G=8's larger initial allocation (8 seqs)
pre-shapes the PyTorch allocator's block pool to match steady-state needs. G=4's smaller
step 0 allocation (reserved=43.35 GiB) fragments, and step 1 backward can't reuse the
reserved blocks despite ~20 GiB l0_free — the gap between allocated (30.40) and reserved
(43.35) is 13 GiB of unusable fragmented blocks.

**Why batch=2 is worse than G=8 (both produce 8 sequences)**: batch=2 stores 2 prompts +
8 completions (different memory layout); G=8 stores 1 prompt + 8 completions. The extra
prompt storage and different allocation pattern pushes batch=2 into OOM during step 0
backward.

**FSDP communication dominance**: GRPO fwd+bwd = 42.8-43.5s for 8 sequences. Of this,
~85% is AllGather/ReduceScatter for 30B total params (~122 GB per fwd+bwd). Only ~5-7s
is actual compute (MoE expert BMM + attention). Expert Parallelism is the only path to
reduce communication volume (from ~122 GB to ~13 GB for attention+router only).

### torch.compile on MoE — Impractical on XPU

torch.compile is not viable for MoE training on XPU due to SYCL compilation overhead:

- **Full model** (48 layers including experts): 500+ SYCL kernel modules compiled over
  25+ minutes. Never finished within 1-hour job walltime. `L0 build module failed` /
  `IGC: Internal Compiler Error` when processes were killed.
- **Attention-only** (with `@torch.compiler.disable` on `GroupedExpertsHF.forward` and
  `GroupedExperts.forward`): 144 kernels compiled before walltime expired. ~75% fewer
  kernels but still too slow — each SYCL C++ kernel requires an `icpx` invocation.

**Root cause**: The XPU inductor backend generates SYCL C++ source that must be compiled
by `icpx`. This is fundamentally slower than Triton's PTX generation on CUDA. For 48
transformer layers × 10 ranks, hundreds of unique compiled graphs are generated. Not a
graph complexity issue — even attention-only compilation is impractical.

**Current state**: `@torch.compiler.disable` decorators remain on both expert forward
methods (`torchtune/models/qwen3_moe/_experts.py`, `torchtune/modules/moe/experts.py`).
These prevent wasted compilation time if `compile=True` is passed, though compile remains
impractical for the non-expert layers too on XPU.

**Recommendation**: Do not use `compile=True` for MoE models on XPU. The 6.3× BMM
speedup from scatter-pad-bmm-gather is the best compute optimization available. Further
gains require Expert Parallelism to cut communication overhead.

Report: `docs/reports/moe_p0p1_experiments_20260427.md`

### Launcher

```bash
# SHM weight sync (production) — use G=8 config
experiments/qwen3_moe/run_grpo_e2e.sh

# P0/P1 experiment tests
experiments/qwen3_moe/run_moe_p0p1_tests.sh

# XCCL weight sync (blocked on single-node)
experiments/wsync/run_moe_xccl_test.sh
```

Config: `recipes/configs/dev/production/qwen3_30b_a3b_grpo_xpu.yaml`

---

## Gemma4 26B-A4B Expert Parallelism (paused at v154; see Qwen3 EP for active work)

Status and implementation notes for Expert Parallelism (EP) on Gemma4 26B-A4B,
EP=4/DP=3 on 12 tiles per node. **EP for Gemma4 is paused** — see "Backward
Dispatch Saga" and "v154 Result" below for the autograd-ordering and router-
determinism blockers. Active EP work is on Qwen3-30B-A3B (later in this doc).

## Goal (revised 2026-04-22, audit)

Original aim was to partition 128 experts × 4 EP → 32 experts/tile, claimed at
~3 GiB/tile savings. The "Memory Model" section below shows that claim does not
hold once non-expert FSDP2 becomes coarser (3-rank vs 12-rank), and the only
end-to-end measurement so far (April 11, batch=1, grpo_samples=2) showed EP=4
**2.6× slower** than EP=1 across every phase (gen 2.2×, grpo 2.4×, opt 24×
because expert AdamW runs on CPU). EP becomes interesting only at batches large
enough to amortize dispatch latency *and* once the autograd/IPC blockers below
are resolved. Until then the headline savings claim is retracted.

Topology: `dp_replicate=3 × dp_shard=4` on 12 tiles. The `dp_shard` submesh
doubles as the EP communicator group — the same 4 ranks handle both FSDP
sharding of non-expert parameters and Expert Parallel token dispatch.

---

## Model Architecture

Gemma4 26B-A4B has 62 transformer layers, 30 of which are MoE. Each MoE layer:

```
Gemma4TransformerLayer
  ├── attn          (Gemma4Attention)      — replicated
  ├── mlp           (dense FFN)            — replicated (non-MoE layers only)
  └── moe_block     (MoE)                  — MoE layers only
        ├── router  (Gemma4MoeRouter)      — replicated
        └── experts (GroupedExperts)       — EP-partitioned
```

Key parameters: 128 experts/layer, top-2 routing. Each expert is a small FFN:
`gate_proj` + `up_proj` + `down_proj` with shapes `(num_experts, dim, hidden_dim)`.
After EP=4 partitioning: `(32, dim, hidden_dim)` per tile.

---

## Files Modified

| File | Change |
|------|--------|
| `torchtune/modules/moe/utils.py` | Added `_permute`, `_unpermute` (pure torch, no Triton) |
| `torchtune/modules/moe/_parallelism.py` | Added `ExpertParallel` (and the now-removed `apply_ep_weight_sharding`; expert weight slicing happens in the recipe pre-FSDP2, see "EP weight slice contract" below) |
| `torchtune/modules/moe/__init__.py` | Exports `ExpertParallel`, `wire_ep_to_moe_modules` |
| `torchtune/modules/moe/experts.py` | Fixed `_forward_no_grouped_mm` to use local expert count |
| `torchtune/models/gemma4/_parallelism.py` | New: `gemma4_ep_plan()` returning EP plan for 30 MoE layers |
| `torchtune/models/gemma4/__init__.py` | Exported `gemma4_ep_plan` |

### Test / validation scripts

| File | Purpose |
|------|---------|
| `experiments/ep_parallelism/test_ep_smoke.py` | 4-subtest smoke test (imports, permute, all-to-all, forward) |
| `experiments/ep_parallelism/test_ep_correctness.py` | EP=2 vs replicated output comparison (bit-exact) |
| `experiments/ep_parallelism/test_ep_model_setup.py` | 12-tile EP=4/DP=3 topology validation with real checkpoint |
| `experiments/ep_parallelism/test_experts_forward.py` | Expert forward kernel sanity |
| `tests/torchtune/dev/rl/test_ep_slice_contract.py` | **CPU-safe regression**: recipe pre-FSDP2 expert slice formula must match `_token_dispatch`'s ownership formula (interleaved). Runs on a login node in seconds. |

Recommended single-node smoke (held compute node, EP=2):

```bash
torchrun --standalone --nproc_per_node=2 \
    experiments/ep_parallelism/test_ep_correctness.py
```

---

## Implementation Design

### Why hook-based EP (not DTensor)

The natural approach is to mark expert parameters as `DTensor` with `Shard(0)` placement
on the EP mesh. This was tried first and fails:

When the outer `shard_model()` calls `fully_shard(layer, mesh=dp_mesh)`, FSDP2 internally
tries to compose mesh dimensions and produces `('dp_replicate', 'dp_shard', 'dp_shard')` —
an invalid 3D mesh from a 2D source.

**Solution**: register plain PyTorch forward hooks instead of using `distribute_module`.
Expert parameters stay as regular `nn.Parameter` tensors (sliced to local shard size).
All-to-All dispatch/combine runs in `register_forward_pre_hook` / `register_forward_hook`.
No DTensor, no mesh composition conflict.

### ExpertParallel (`torchtune/modules/moe/_parallelism.py`)

Registered via `parallelize_module(model, ep_mesh, ep_plan)`. For each `GroupedExperts`
module it:

1. Stores `module._ep_device_mesh = device_mesh` for later weight slicing
2. Registers a forward pre-hook (`_token_dispatch`) and forward hook (`_token_combine`)

**`_token_dispatch`** (pre-hook):
1. All-gather the per-rank `num_tokens_per_expert` histograms across the EP group to build
   the full `(ep_degree, num_experts)` dispatch matrix
2. Compute `input_splits` (tokens this rank sends to each EP peer) and `output_splits`
   (tokens this rank receives from each EP peer)
3. `all_to_all_single_autograd`: route tokens to expert-owning ranks
4. `_permute`: reorder received tokens from source-rank-major to local-expert-major order

**`_token_combine`** (post-hook):
1. `_unpermute`: scatter expert outputs back to source-rank-major order
2. `all_to_all_single_autograd` (swapped splits): return processed tokens to originating ranks

Gradient flow is automatic via `all_to_all_single_autograd`.

### EP weight slice contract (`recipes/dev/grpo_full_finetune_distributed_xpu.py`)

Expert weight slicing is performed in the recipe **before** `load_from_full_model_state_dict`,
operating on the full-precision tensor straight from the HF checkpoint. The slice formula
must match `_token_dispatch`'s expert-ownership formula (interleaved):

```python
# Recipe (pre-FSDP2): rank r owns experts r, r+ep_degree, r+2*ep_degree, ...
model_sd[_sd_name] = _ft[_ep_rank::_ep_degree].contiguous()

# Dispatch (_token_dispatch): for local expert i, global expert g = ep_rank + i*ep_degree
g = ep_rank + local_exp_idx * ep_degree
```

**Critical**: any contiguous slice (`_ft[r*n_local : (r+1)*n_local]`) silently permutes
tokens against weights and produces meaningless training. The CPU-safe regression
`tests/torchtune/dev/rl/test_ep_slice_contract.py` enforces both formulas agree across
several `(num_experts, ep_degree)` pairs.

The previous `apply_ep_weight_sharding` helper (which performed the same slicing post-load)
has been removed. Pre-load slicing in the recipe is preferred because the meta-param shape
is set to `(n_local, dim, dim)` before `load_from_full_model_state_dict` runs, so FSDP2's
shape check passes naturally.

### _permute / _unpermute (`torchtune/modules/moe/utils.py`)

Pure torch implementation (no Triton). Builds a permutation index that reorders tokens
from the source-rank-major layout produced by All-to-All into the expert-major layout
required by `GroupedExperts.forward()`:

```
After All-to-All:   [ep0_exp0_toks, ep0_exp1_toks, ..., ep1_exp0_toks, ...]
After _permute:     [exp0_from_all_ranks, exp1_from_all_ranks, ...]
```

Compatible with XPU's loop-based expert path (no `grouped_mm` padding required).

### experts.py fix

`GroupedExperts._forward_no_grouped_mm` used `self.num_experts` (global count = 128) as
the loop bound `E`. After EP weight slicing, `gate_proj` has only `num_experts / ep_degree`
rows. Fix: use `E = num_tokens_per_expert.shape[0]` (local count = 32 for EP=4).

### gemma4_ep_plan (`torchtune/models/gemma4/_parallelism.py`)

Iterates `model.layers` and maps each MoE layer's `experts` to `ExpertParallel()`:

```python
ep_plan = gemma4_ep_plan(model)      # 30 entries: layers.{i}.moe_block.experts
parallelize_module(model, ep_mesh, ep_plan)
```

Note: `model.layers` is a `ModuleList`, not `ModuleDict` — iterate with `enumerate`,
not `.values()`.

---

## Setup Sequence (current as of v158+)

The original recipe in this section called `apply_ep_weight_sharding` after
`load_state_dict`. That has been replaced. The current path (recipe lines
~1815-2240 in `recipes/dev/grpo_full_finetune_distributed_xpu.py`) is:

```python
# 1. Build 2D mesh
dp_mesh = init_device_mesh(device.type, (dp_replicate, dp_shard),
                            mesh_dim_names=("dp_replicate", "dp_shard"))
ep_mesh = dp_mesh["dp_shard"]

# 2. Eager process group initialization (prevents XCCL deadlock in forward hooks)
_ = ep_mesh.get_group()
_ = dp_mesh.get_all_groups()
dist.barrier()

# 3. Build model on meta device
with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = gemma4_26b_a4b()  # or qwen3_30b_a3b

# 4. Pre-FSDP2 meta-param shrinking: replace expert nn.Parameter shapes on the
#    meta model so each rank only materializes its EP shard. The full-checkpoint
#    state_dict is then **pre-sliced** at load-time (model_sd[..., ep_rank_slice])
#    instead of being loaded full and then sharded post-hoc. This sidesteps the
#    pre-shard memory peak that killed earlier versions.
#    (Code: `_shrink_ep_meta_params()` and the model_sd pre-slice block in the
#     recipe init flow.)

# 5. Apply EP hooks (registers _token_dispatch / _token_combine pre/post hooks)
ep_plan = gemma4_ep_plan(model)            # or qwen3_moe_ep_plan(model)
parallelize_module(model, ep_mesh, ep_plan)

# 6. Stage checkpoint to /tmp (rank 0 copies, barrier, all ranks load)
if rank == 0:
    shutil.copy2(LUSTRE_CKPT, TMP_CKPT)
dist.barrier()

# 7. Materialize model on XPU and load the (pre-sliced) checkpoint
model.to_empty(device=device)
sd = checkpointer.load_checkpoint()
model.load_state_dict(sd["model"], strict=False)

# 8. Layered FSDP2:
#    - Non-expert params: fully_shard on dp_replicate (3 ranks),
#      reshard_after_forward=False (ZeRO-2). Disjoint from ep_mesh ranks.
#    - Expert params: trivial 1-rank "solo" FSDP2 with reduce_grads=False
#      (pure local wrapping; no comm). Silences ze_handle_manager crashes
#      on 1-rank reduce_scatter.
#    - All FSDP2 groups have reduce_grads=False (suppressed). Gradient sync
#      is done post-backward by `_ep_post_backward_grad_sync_xccl()` over
#      an explicit `_XCCL_DP_REP_PG`.

# 9. EP collectives (AG/RS/NTPE-AG) use **gloo CPU-bounce** (`_GLOO_EP_PG`,
#    a per-EP-group gloo PG separate from `_GLOO_DP_SHARD_PG`). XCCL is NOT
#    used in the EP dispatch path; see _ep_all_gather/_ep_reduce_scatter in
#    `torchtune/modules/moe/_parallelism.py`.
```

`apply_ep_weight_sharding` (post-load slicing) was removed; the pre-FSDP2
meta-param + state_dict pre-slice in the recipe replaces it. See the
"EP weight slice contract" section above for the active formula and the
CPU-safe regression test (`test_ep_slice_contract.py`) that guards it.

---

## XCCL / Environment Requirements

For jobs that create multiple process groups (2D mesh → submesh groups):

```bash
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_MODE=pmi
export CCL_KVS_IFACE=hsn0          # required for submesh group creation
export FI_PROVIDER=cxi              # NOT shm — shm fails for nested groups
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

Also required: bypass `torchtune.__init__` to prevent XCCL USM pointer corruption
(torchtune imports torchao which corrupts XCCL's USM table on Aurora):

```python
# Must happen before any torch or torchtune import
import types, importlib.util, sys
if "torchtune" not in sys.modules:
    spec = importlib.util.find_spec("torchtune")
    pkg = types.ModuleType("torchtune")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__file__ = ...
    sys.modules["torchtune"] = pkg
```

---

## Validation Status (setup-phase only — see v141+ sagas for end-to-end)

These tests cover **module-level setup correctness**, not end-to-end training.
End-to-end EP=4/DP=3 GRPO training is **not** validated on Gemma4 (paused at
v154 on ScatterAddBackward0 router-determinism). For Qwen3 EP=4/DP=6 (3-node):
**first `loss=` line reached at v8i** (G=1/fbs=1/NSTEPS=1). The v59
reduce_grads architectural issue was then unblocked at EP=8 / dp_replicate=1
on a 2-node hold by adding the v9 `_ep_release_fsdp_unsharded_grads` helper
plus `AdamWBf16` — see the "Qwen3-30B-A3B EP=8 v10 series" section below
and `docs/reports/qwen3_ep_v10_20260430.md`. The "Backward Dispatch Saga"
and the v8/v10 sections are the authoritative status for any EP work.

| Test | Config | Status | Notes |
|------|--------|--------|-------|
| Import smoke test | EP=2, 2 tiles | **PASS** | `ExpertParallel`, `_permute`, `all_to_all_single_autograd` |
| Permute/unpermute | EP=2, 2 tiles | **PASS** | Round-trip, all token counts |
| All-to-All forward | EP=2, 2 tiles | **PASS** | Correct shapes and values |
| Import smoke test | EP=4, 4 tiles | **PASS** | |
| All-to-All forward | EP=4, 4 tiles | **PASS** | |
| EP=2 vs replicated | EP=2, 2 tiles | **PASS** | Bit-exact match (max_err=0.0) |
| 12-tile forward pass | EP=4/DP=3, 12 tiles | **PASS (setup only)** | See results below |
| Gemma4 EP=4/DP=3 GRPO end-to-end | 12 tiles | **BLOCKED** | v154 ScatterAddBackward0 ±1; see Backward Dispatch Saga |
| Qwen3 EP=4/DP=3 GRPO end-to-end (2-node 12+12, v1-v7) | 12 train tiles | **BLOCKED** | train-fwd PDE; never reached `loss=` on this topology |
| Qwen3 EP=4/DP=6 GRPO end-to-end (3-node, v8 series) | 24 train tiles | **PARTIAL** | v8i first `loss=` at G=1/fbs=1/NSTEPS=1 rc=0; G≥2 and NSTEPS≥2 blocked by v59 reduce_grads |
| Qwen3 EP=8/dp_replicate=1 GRPO end-to-end (2-node, v10 series) | 8 train tiles | **PASS** | v10f G=4 NSTEPS=3 FBS=1 + v9 helper + AdamWBf16: 3/3 steps clean, finite loss, ~200s/step. PRE-STEP-N flat. |
| EP=1 (replicated) vs EP=4 throughput | batch=1, G=2 | **EP=4 is 2.6× SLOWER** | April 11; see new throughput section above |

### 12-tile forward pass results (test_ep_model_setup.py)

Node: `x4117c4s3b0n0`, job 8431784

```
[Rank 0] All process groups initialized
[Rank 0] ExpertParallel applied to 30 modules
[Rank 0] load_state_dict: 0 missing, 0 unexpected
[Rank 0] Checkpoint loaded in 357.6s          ← from Lustre; /tmp staging now added (~62s)
[Rank 0] Peak memory after load (pre-slice): 47.12 GiB
[Rank 0] EP weight sharding applied to 30 expert modules
[Rank 0] Peak memory after EP slice: 47.12 GiB  ← XPU allocator holds pool; actual RSS reduced
[Rank 0] Forward pass OK in 2.07s, out shape: torch.Size([1, 512, 262144])
[Rank 0] Peak memory after forward: 47.62 GiB
[Rank 0] === EP=4/DP=3 Model Setup Test: PASS ===
```

Memory note: peak stays at 47 GiB because the XPU standalone allocator retains freed
blocks in its pool rather than returning them to the OS. The actual expert parameter
resident set did shrink 4× (32 experts vs 128); the benefit will show in training where
optimizer states are the dominant factor.

---

## FSDP2 + EP Communicator Layering (resolved)

The earlier section claimed FSDP2 was "skipped" — that workaround is **gone**.
The recipe now uses a layered scheme that prevents the XCCL group-conflict
without disabling FSDP2:

- **Non-expert params**: FSDP2 on `dp_replicate` (3 ranks),
  `reshard_after_forward=False`. The shard group is disjoint from the `ep_mesh`
  ranks, so FSDP AllGather and EP collectives never share a communicator.
- **Expert params**: trivial 1-rank "solo" FSDP2 with `reduce_grads=False` —
  pure local wrapping, no inter-rank comm. Used only to silence the
  `ze_handle_manager` crash that 1-rank reduce_scatter hits on Aurora.
- **All FSDP2 `reduce_grads` are suppressed.** Gradient sync runs after backward
  via `_ep_post_backward_grad_sync_xccl()` on an explicit `_XCCL_DP_REP_PG`.
- **EP collectives use gloo CPU-bounce.** `_GLOO_EP_PG` is a per-EP-group gloo
  PG, separate from `_GLOO_DP_SHARD_PG` (which carries FSDP grad sync). Sharing
  one gloo communicator caused sequence-number collisions at op #259 (v152).
  See `torchtune/modules/moe/_parallelism.py` `_ep_all_gather`/`_ep_reduce_scatter`
  and the v152→v153 entries below.

This layering eliminates the XCCL communicator-conflict mode that motivated
the original "skip FSDP2" workaround. The remaining EP blockers (op #259
desync, ScatterAddBackward0 mismatch, train-fwd PDE) are autograd / IPC
issues, not FSDP2 conflicts.

---

## Memory Model — net-neutral, not net-positive

Earlier versions of this section claimed EP=4 saves ~3 GiB/tile vs EP=1. The
v17/v40 runtime data and the April 11 EP=1 vs EP=4 benchmark contradict that
at the achieved configuration.

Without any FSDP2 (initial setup-only test state):

| Component | EP=1 | EP=4 (no FSDP2) |
|-----------|------|-----------------|
| Expert params | ~4.0 GiB | ~1.0 GiB |
| Non-expert params | ~16.5 GiB | ~16.5 GiB |
| **Total params** | ~20.5 GiB | ~17.5 GiB |

With FSDP2 on `dp_replicate` (3-rank) for non-expert params (training target):

| Component | EP=1 / FSDP2 12-rank | EP=4 / FSDP2 3-rank |
|-----------|----------------------|---------------------|
| Expert params | ~4.0 GiB | ~1.0 GiB |
| Non-expert FSDP shard | ~1.4 GiB (÷12) | ~5.5 GiB (÷3) |
| **Total params** | ~5.4 GiB | ~6.5 GiB |
| Optimizer states (Adam) | ~10.8 GiB | ~13.0 GiB |
| Activations (training) | ~4.5 GiB | ~4.5 GiB |
| **Estimated peak** | ~20.7 GiB | ~24.0 GiB |

The 3 GiB/tile expert savings are consumed by +2.1 GiB coarser non-expert FSDP
and +4.2 GiB expert optimizer state on the EP path. Net memory at the achieved
configuration is **roughly neutral** — not the headline savings the earlier
section advertised. EP only wins memory once the configuration enables a batch
size EP=1 cannot fit; that crossover has not yet been measured.

---

## EP=1 vs EP=4 throughput (April 11 benchmark — strictly worse at small batch)

This finding lived only in `project_ep_implementation.md`; surfaced here so it
isn't missed by anyone reading the doc to plan EP work.

At batch=1, grpo_samples=2 (before the BWD blockers resurfaced):

| Phase | EP=1 | EP=4 |
|-------|------|------|
| gen | 60.2s | 131s (2.2×) |
| grpo | 31.2s | 74s (2.4×) |
| opt | 1.6s | 39s (24× — expert AdamW on CPU) |
| **total** | **93s** | **246s (2.6×)** |

EP=4 is **strictly worse** at small batches, end-to-end. The 24× opt-phase
penalty alone dwarfs any communication savings. EP only becomes interesting
once the batch is large enough that expert GEMM dominates dispatch latency
**and** AdamW for the expert shards can move back to GPU. Neither has been
measured. "≤+10% step time" from the earlier "What's Next" target was based
on no data — disregard until a fresh benchmark exists.

---

## What's Next

1. **~~Unblock Qwen3 EP G≥2 / NSTEPS≥2~~ — DONE 2026-04-30 (v10 series).**
   The v9 `_ep_release_fsdp_unsharded_grads` helper materializes sharded
   grads after each chunk (gloo CPU-bounce reduce-scatter, behind a global
   barrier) and the per-step defensive sweep at line 3322 catches any
   residue. With plain `torch.optim.AdamW` the helper alone was not enough:
   fp32 momentum+variance materialized ~22.8 GiB on the first step and
   pinned PRE-STEP-1 to 29.99 GiB. `torchtune.dev.bioreason.optim.AdamWBf16`
   (CPU-side state, peak GPU = single param delta) restored ~14 GiB and let
   v10f run G=4 NSTEPS=3 FBS=1 to completion (rc=0). **Cost is steep**:
   ~200s/step, with ~99s in the helper and ~86s in the CPU optimizer; only
   ~10s of the step is actual fwd+bwd compute. See "Qwen3-30B-A3B EP=8 v10
   series" below and `docs/reports/qwen3_ep_v10_20260430.md`. **Resolve the
   Gemma4 EP backward blocker** (router-determinism in AC recompute, v154)
   is still open as a parallel research task.
2. **Re-benchmark EP=1 vs EP=4 at a batch size large enough to amortize
   dispatch** — current data only covers grpo_samples=2 where EP=4 is 2.6×
   slower. Find the crossover or confirm there isn't one on this hardware.
3. **Move expert AdamW back to GPU** if memory allows. The 24× opt-phase
   penalty in the EP=4 column above is entirely CPU optimizer overhead.
   On v10f this is still the dominant pattern at EP=8: ~86s/step in
   AdamWBf16 vs <10s of fwd+bwd compute. Headroom on EP=8 dp_replicate=1
   is ~7 GiB at PRE-STEP — not enough to fit fp32 AdamW state on-device.
   3-node dp_replicate=3 (untested) would split optimizer state across
   replicas and may restore the on-device path.

---

## Qwen3-30B-A3B EP=8 v10 series — multi-step unblocked (2026-04-30)

PBS jobs 8460378 and 8461394 (capacity, 2 nodes). Single training node,
EP=8 / dp_replicate=1 / dp_shard=8 (8 of 12 tiles), with the 3rd-node
vLLM 3×TP=4 server. Recipe surgery near
`grpo_full_finetune_distributed_xpu.py:547` builds a 1D `dp_shard` mesh
when `self._dp_mesh is None` and routes the v9 helper's PG map.

### Result matrix

| Run    | Config                                    | PRE-STEP-0 | PRE-STEP-1 | Outcome |
|--------|-------------------------------------------|------------|------------|---------|
| v10a   | G=1 NSTEPS=1, plain AdamW                 | 7.19 GiB   | n/a        | PASS (rc=0); loss=nan G=1 0/0 advantage artifact |
| v10b   | G=2 NSTEPS=1 fbs=2 (single chunk B=2)     | 7.19       | n/a        | banned:1 PDE step-0 fwd; activation OOM at B=2 |
| v10b2  | G=2 NSTEPS=1 FBS=1 (real 2 chunks)        | 7.19       | n/a        | **PASS**; loss=0.0061; chunked accumulate path validated |
| v10c   | G=4 NSTEPS=3 FBS=1                        | 7.19       | 29.99      | step-1 PASS, step-2 fwd OOM (l0_free=0.05 GiB) |
| v10d   | G=1 NSTEPS=3 FBS=1                        | 7.19       | 29.96      | step-1 PASS, step-2 fwd OOM same as v10c — proves AdamW state is the wall |
| v10e   | G=2 NSTEPS=2 FBS=1, **AdamWBf16**         | 7.19       | **15.77**  | **PASS rc=0**; both steps finite loss; ratios=1.000 |
| v10f   | G=4 NSTEPS=3 FBS=1, **AdamWBf16**         | 7.19       | 15.77 / 15.47 | **PASS rc=0**; ~3:20/step (192-204s); per-step sweep confirmed; loss 0.006-0.009 finite |

### What we learned

- **v9 helper works at EP=8.** All 97 FSDPParamGroups release through the
  gloo dp_shard PG (~25s per release). The v59 silent no-op is fully
  replaced.
- **AdamW fp32 state is the second wall.** v10c and v10d (same crash at
  G=1 minimum activation) confirm the +22.8 GiB PRE-STEP-1 jump is fp32
  momentum+variance, not unfreed grads or activation pressure.
- **AdamWBf16 unblocks multi-step.** PRE-STEP-1 dropped 29.99 → 15.77 GiB,
  restoring ~14 GiB of L0 headroom. The per-step defensive sweep at line
  3322 also kept PRE-STEP-2 below PRE-STEP-1 in v10f.

### Per-step wall-clock (v10f, averaged over 3 steps)

| Phase | Time | Share |
|------|------|------|
| gen (vLLM HTTP)        | ~5.6 s   | 3% |
| grpo (fwd + bwd + helper) | ~108.9 s | 54% |
| &nbsp;&nbsp;– fwd      | ~3.8 s | |
| &nbsp;&nbsp;– bwd      | ~5.6 s | |
| &nbsp;&nbsp;– **v9 release helper** | **~99 s** | |
| clip                   | ~0.1 s | <1% |
| optimizer (AdamWBf16)  | ~85.8 s | 43% |

~93% of step time is CPU-side workarounds (gloo bounce + CPU optimizer).
This is **3.6× slower** than the validated EP=1 path (54.8s/step at G=8,
2026-04-27) at half the rollout count. Two paths can recover much of the
gap, neither yet committed:

A. **3-node EP=8 dp_replicate=3 / dp_shard=8.** Tests whether v75 XCCL
   cross-replica sync works with the real grads the helper now produces,
   and whether spreading optimizer state across 3 replicas restores
   enough headroom to use plain AdamW back on device.
B. **Rewrite `_ep_release_fsdp_unsharded_grads` to use XCCL reduce-scatter
   on-device.** The helper currently does CPU gloo bounce because the v59
   race was originally diagnosed in XCCL. Re-test whether the global
   barrier alone is sufficient ordering for on-device XCCL.

Production envelope: optimizer **must be**
`torchtune.dev.bioreason.optim.AdamWBf16`; `forward_batch_size=1` is
required to exercise the chunked-accumulate path under `G≥2`. Validated
to `G=4 NSTEPS=3` with finite losses and PRE-STEP flat. Loss quality is
still in the regime where many responses get truncated before any reward
signal — a real reward curve has not been run.

### Memories and launchers

- `memory/project_qwen3_ep_v10_unblocked.md` — full result matrix
- `memory/project_qwen3_ep_v9_helper.md` — helper design
- `memory/project_qwen3_ep_v8_3node.md` — v59 unified blocker
- `experiments/ep_parallelism/hold_qwen3_ep8_v10{a,b,b2,c,d,e,f}.sh`
- `recipes/dev/run_qwen3_30b_ep8_vllm_2node.sh`

---

## Backward Dispatch Saga (v141–v153, April 22, 2026)

After abandoning AllToAll (which deadlocked v18–v136 with XCCL SIGSEGV in backward),
EP dispatch was reformulated as **AllGather + ReduceScatter** (Mula paper, arXiv 2604.00785):

```
Forward:   AllGather tokens → index-select for local experts → expert GEMM
                            → scatter_add → ReduceScatter
Backward:  automatic via PyTorch autograd (AG ↔ RS swap)
```

This unblocked the SIGSEGV but produced a new deterministic deadlock: **op #259 RS-BWD**.

### Op structure (300 EP ops/step)

With AC wrapping `TransformerSelfAttentionLayer` (full layer including MoE) and
`use_reentrant=True`, each MoE layer contributes 4 BWD ops: AG-FWD recompute,
RS-FWD recompute, AG-BWD, RS-BWD. Total per step:

| Pass | Range | Notes |
|------|-------|-------|
| Policy logprob FWD | 0–59 | `no_grad`; 30 layers × 2 ops |
| Ref logprob FWD | 60–119 | `no_grad`; ref model also wired with EP |
| Training FWD | 120–179 | 30 layers × 2 ops |
| Training BWD | 180–299 | 30 layers × 4 ops (AC recompute + grad) |

Op #259 = RS-BWD for layer 10 (10th layer from front, 19th-from-back in BWD order).

### v141–v152: gloo CPU-bounce migration

To avoid XCCL deadlock in the AC-recompute interleaving, all EP collectives were
migrated to gloo CPU-bounce (move tensor to CPU, gloo all_reduce, slice, move back):

| Version | Change | Result |
|---------|--------|--------|
| v141 | Native XCCL `_c10d_reduce_scatter` (bypass gloo monkey-patch) | All 260+ ops complete; needs full re-run for confirmation |
| v143 | `dist.barrier(ep_pg)` after each XCCL collective (diagnostic) | All ops complete; barriers masked the bug + drained OFI CQ |
| v144 | Remove diagnostic barriers | Immediate failure — confirms diagnostic was load-bearing |
| v148–v151 | Pure gloo CPU-bounce for AG and RS | All FWD ops complete; **op #259 RS-BWD timeout** |
| v152 | Dedicated `_GLOO_EP_PG` (separate gloo group from FSDP2 grad sync) | Same op #259 RS-BWD deadlock; eliminated FSDP2 contention as cause |

**v152 forensics (job 8444929):** all 12 ranks complete ops 0–258. At op #259:
6 ranks print `ENTER RS-BWD`, 0 print `COLL-DONE`. The other ~6 ranks (one per
EP group, plus some receivers) never call op #259 at all. Between ops 258 and 259
there is **only local computation** — no inter-rank collective. So either ranks
crashed silently between ops, or the gloo TCP connection state itself was corrupted.

### v153: GLOO_SOCKET_IFNAME=lo + zero XCCL in EP path (job 8445116)

**Hypothesis**: gloo TCP defaults to the CXI HSN NIC on Aurora. XCCL also uses the
CXI NIC via OFI. After 256+ ops of heavy XCCL traffic, the CXI NIC's OFI completion
queue accumulates stale events that contaminate gloo TCP connection state. Some ranks
lose the connection silently and never call op #259 → ring deadlock.

**Changes:**
1. `GLOO_SOCKET_IFNAME=lo` — force gloo TCP onto kernel loopback (127.0.0.1),
   bypassing the CXI NIC entirely. All 12 training tiles are on one node, so
   loopback is correct.
2. Replace the last XCCL call in EP dispatch (NTPE histogram all-gather between
   AG-FWD and RS-FWD) with gloo CPU-bounce. **Zero XCCL in EP dispatch path.**
3. Per-rank logging (`[rank{r}]` prefix) on every EP op + new `PRE-RS-BWD` print
   in `_AllGatherRS.backward`.
4. EP gloo group timeout reduced to 120s (was 1800s) for faster diagnosis.

**Result:** different failure mode at the same op #259.

- **All 12 ranks now reach op #259** (loopback fix solved the "missing rank" problem).
- But rank 8 **crashes hard with exitcode 1** during the gloo all_reduce
  (other ranks were SIGTERM'd, exit -15).
- Loopback peers print `Connection closed by peer [127.0.0.1]:<port>` —
  consistent with rank 8's process dying mid-collective and dropping its TCP socket.
- Other ranks then time out at 120s waiting for send/recv that will never complete.

**The new clue (rank desync at op #258):** with per-rank logging on, a structural
problem became visible. At op #258, ranks split into two groups:

| Ranks | EP-OP #258 label |
|-------|------------------|
| 0, 2, 3, 4, 6, 7, 8, 10, 11 | `AG-BWD` |
| **1, 5, 9** | `AG-FWD` (forward recompute of NEXT layer) |

Ranks 1, 5, 9 are the **local-index-1 rank within each EP group**
([0,1,2,3], [4,5,6,7], [8,9,10,11]). They are exactly **one op behind** the others.

`_EP_OP_N` is a per-rank Python counter — each rank increments it on every collective
it calls. Gloo matches collectives by **call order**, not by op label. So a same-numbered
op on different ranks within the same EP group can refer to **different layers**.
Once one rank in a group is out of step, every subsequent collective in that group
mismatches: ranks send a tensor sized for layer N while their peer expects layer M.

This explains everything:
- Why op #259 is the failure point: 19 prior matched gloo ops worked because the
  routing distribution happened to produce compatible sizes; eventually a mismatch
  in tensor shape or content produces an unrecoverable state.
- Why one rank crashes hard (rank 8 exitcode 1): a size-mismatched gloo op asserts
  or segfaults on one process before the others.
- Why the v152 "missing ranks" pattern matched the same local-index-1 ranks
  (~ranks 3, 5, 9) — the desync in v152 manifested as never-arriving ranks; in v153
  with the timing-shifted loopback transport, it manifests as a hard crash mid-collective.

### Root cause — autograd execution order is not deterministic per rank

PyTorch's autograd engine schedules backward hooks in a topological order determined
by the operation order seen during forward. With AC + `use_reentrant=True`, the forward
recompute happens during backward. **Subtle differences in tensor allocation order or
hook registration can cause autograd to schedule layers in different orders on different
ranks** — even when the model and inputs are identical. The MoE forward hook
(`_token_dispatch`/`_token_combine`) registration order interacts with PyTorch's
autograd sort, and the result is rank-dependent.

When backward ops are not lock-step across ranks within a collective group, the global
op counter desyncs and gloo collectives mismatch.

### Why this is hard to fix

The fix cannot be "set GLOO_SOCKET_IFNAME=lo" or "use a different transport" — those
are timing band-aids that will eventually expose the same desync. The real fix requires
one of:

1. **Tag/identify each collective explicitly** instead of relying on call order.
   Gloo doesn't support tagging well. NCCL/XCCL with `barrier()` after each op would
   force lock-step but kills throughput.
2. **Force deterministic backward order** via `torch.autograd.backward(... retain_graph=True)`
   with explicit per-layer backward calls instead of a single `loss.backward()`.
   Major recipe surgery.
3. **Avoid AC for MoE layers** — `use_reentrant=False` AC, or no AC at all on layers
   with EP. Memory cost: ~30 layers × activations / layer; may not fit at G=4.
4. **Move EP collectives outside the AC region** — pre-dispatch all tokens before
   the layer's forward, run experts as a side effect, post-combine after. Decouples
   collective ordering from autograd scheduling.

### Recommendation (revised)

EP for Gemma4 26B-A4B is blocked on a deeper autograd-ordering issue, not a transport
issue. After 153 versions, the path forward is:

- **Short term**: keep EP as a research effort. Production training continues with
  EP=1 (replicated experts) using the proven 12-tile or 2-node HSDP recipes.
- **Next experiment**: option 4 above — restructure `_token_dispatch`/`_token_combine`
  to run as side-effecting operations outside autograd, with manual gradient handling.
  This is the cleanest decoupling and is reusable for any MoE+AC combination.
- **Fallback**: option 3 — disable AC for the 30 MoE layers and accept the memory hit;
  measure whether G=2 still fits.

---

## Critical Review (2026-04-22, post-v153, no new tests)

After 150+ versions the doc had absorbed claims as fixed truth that don't survive a
re-read of the current code, the v153 logs, and `MEMORY.md`. This section calls those
out and proposes experiments before the next compute spend.

### 1. The doc no longer matches the code

**"FSDP2 + EP Communicator Conflict — Current workaround: FSDP2 is skipped"** (lines 254–264) —
stale. The recipe (`grpo_full_finetune_distributed_xpu.py:1934–2120`) has, for many
versions, used a layered scheme:

1. Non-expert params: FSDP2 on `dp_replicate` (3 ranks), `reshard_after_forward=False`
   (ZeRO-2). Disjoint from `ep_mesh` ranks → no XCCL group conflict.
2. Expert params: trivial 1-rank "solo" FSDP2 with `reduce_grads=False`. Pure local
   wrapping; no comm. Used to silence `ze_handle_manager` crashes on 1-rank RS.
3. All FSDP2 `reduce_grads` are suppressed; gradient sync is done post-backward by
   `_ep_post_backward_grad_sync_xccl()` over an explicit `_XCCL_DP_REP_PG`.
4. EP collectives use **gloo CPU-bounce** (`_GLOO_EP_PG`), not XCCL.

This is materially different from anything described in §"Setup Sequence" or §"Memory
Model". The doc still implies `apply_ep_weight_sharding` is called post-load — the
recipe replaced that with **pre-FSDP2 meta-param shrinking + model_sd pre-slicing**
(v41/v42 lines 1865–1902, 2160–2189). Anyone reading just the doc to understand the
system is missing the core of the actual implementation.

### 2. The "300 ops/step" arithmetic is suspect

Doc says: 30 MoE layers × 2 ops × 5 passes = 300 ops; "180–299: training BWD (4 ops/layer)".
Verify: 30 × 4 = 120 BWD ops → 180 + 120 = 300 ✓. But the **NTPE all-gather** added in
v153 (`_parallelism.py:528–536`) is **not counted** by `_EP_OP_N`. So per-layer dispatch
is actually AG-FWD + NTPE-AG + RS-FWD with two collectives matched against one
counter. The diagnostic logs are misleading by exactly one collective per dispatch.

**Implication**: when the v153 forensics say "ranks 1, 5, 9 are one op behind", the
gloo `_EP_OP_N` they see may be off by an integer multiple of NTPE-AG calls (which all
ranks fire but don't print). The "AG-FWD vs AG-BWD at op 258" finding is real (the
labels are different, not just the counters), but the broader claim of "exactly one
op behind" needs to be re-derived counting NTPE.

### 3. "Per-rank autograd ordering non-determinism" is plausible but unproven

The doc's root-cause story is: AC + `use_reentrant=True` reorders backward execution
on different ranks, so `_EP_OP_N` (a per-rank Python counter) goes out of step within
an EP group, gloo matches collectives by call order, and we eventually crash.

What we actually know from the log:

- Within EP group {0,1,2,3}: at op #258 ranks 0,2,3 are at AG-BWD, rank 1 is at AG-FWD.
- Same pattern in EP groups {4,5,6,7} and {8,9,10,11} — local-index-1 is always the
  outlier. **This is a structural pattern across groups, not random per-rank scheduling.**
- Earlier (op #218 etc.) some ranks complete in 4 enters not 6 (parsed `uniq -c`),
  suggesting partial events were lost or interleaved differently across logs.

A fundamentally non-deterministic explanation should not produce the *exact same*
per-rank pattern across three independent EP groups. Candidate root causes and
audit results (2026-04-22):

1. **A local-rank-dependent code path** firing an extra Python-level collective on
   rank-1 only — **RULED OUT by code audit.** Searched `_parallelism.py`, `moe.py`,
   `experts.py`, `gemma4/_moe.py`, `gemma4/_parallelism.py`, and the EP wiring block
   in `grpo_full_finetune_distributed_xpu.py:1815–2240`. No `if ep_rank == X` or
   `if local_rank == X` branches alter the collective sequence.
2. **Routing-imbalance feedback loop** (still open, now the leading hypothesis):
   per-rank `routed_input.shape[0]` differs each step. Under reentrant AC, the
   *order* in which the recompute pass evaluates submodules of a `TransformerSelfAttentionLayer`
   could depend on tensor sizes registered with autograd, causing a different
   call-order of `_token_dispatch`/`_token_combine` on the rank with the smallest
   batch (which, given the v110 interleaved routing fix, would consistently be the
   *same* local index — explaining the local-index-1 pattern across all three groups
   without invoking randomness). This is the only candidate consistent with both the
   structural pattern and the determinism of `_EP_OP_N` mismatch.
3. **`_ag_gather_idx` instance-cache aliasing** under reentrant AC — **UNLIKELY**
   per code audit. `gemma4/_parallelism.py:49` constructs a fresh `ExpertParallel()`
   per layer (`gemma4_ep_plan`), so each MoE layer has its own
   `self._ag_gather_idx`/`self._ag_s_local`. Cross-layer aliasing is not possible.
   Within-layer recompute could in principle overwrite the attr between FWD-recompute
   of `_token_dispatch` and BWD of `_token_combine`, but `_AllGatherRS.backward`
   captures `gather_idx` from the saved tensors at FWD time, not from `self.*`.

(1) is closed; (3) is closed by construction. **The remaining open root-cause story
is (2) — routing-imbalance interacting with reentrant-AC scheduling.** This is a
narrower target than the doc's previous "autograd ordering is non-deterministic"
framing.

### 4. Things the doc treats as load-bearing that may not be

- **CPU-bounce for AG/RS**: 4× bandwidth tax, kept "to avoid OFI CQ deadlock". Memory
  `project_ccl_naive_bypass_failed.md` documents L0 IPC ALWAYS being used for
  intra-node XPU↔XPU; gloo CPU-bounce dodges L0 IPC entirely. Whether dense Gemma4
  (no EP) gets away with XCCL RS/AG is an interesting datapoint, but the EP path
  has *additional* IPC pressure (12 expert weight handles per layer × 30 layers) that
  may push it past whatever margin dense FSDP2 enjoys. **Note**: my earlier draft of
  this section suggested setting `XPU_USM_ALLOC_SO=usm_caching_alloc.so` to dodge the
  pluggable-allocator IPC bug — that is **wrong at this scale** per
  `feedback_alloc_conf_env_var.md`: the arena allocator has the same IPC
  sub-allocation crash as `expandable_segments` for 26B+ models, and the launcher
  correctly does `unset PYTORCH_ALLOC_CONF` and leaves `XPU_USM_ALLOC_SO` unset. The
  CPU-bounce architecture is therefore **necessary**, not optional, until the
  underlying L0/CCL IPC bug is fixed upstream. Don't try to remove it.
- **`use_reentrant=True`**: chosen v114 specifically because non-reentrant AC was
  triggering a SIGSEGV in the **AllToAll** path. Since AllToAll is gone (replaced by
  AG+RS), the v114 reason for reentrant AC no longer applies. Switching to
  `use_reentrant=False` is independently worth re-trying — it eliminates the *entire*
  category of "FWD recompute interleaved with BWD grad" issues that the doc fingers.
- **NTPE all-gather is needed at all**: AG-FWD already gives every rank the full
  `(EP*S, dim)` tensor. The NTPE collective only exists to compute the gather indices.
  These can be derived locally from the routed token positions (which are themselves
  expert-sorted on each source rank) if we standardize the per-rank S and a known
  cumsum scheme. Removing NTPE drops one gloo collective per layer and one source of
  desync.

### 5. Memory model is internally inconsistent

The two tables in §"Memory Model" disagree: the "without FSDP2" table says EP=4 saves
3 GiB on expert params; the "with FSDP2" table says EP=4 *peaks higher* than EP=1
because non-expert FSDP becomes coarser (3-rank vs 12-rank). Reconciled by the v17/v40
runtime data (project memory `project_ep_implementation.md`): EP=4 step 0 reaches
~12 GiB pre-step; the "memory savings" claim in §Goal (1 GiB/tile vs 4 GiB/tile)
is consumed by the +2.1 GiB of coarser FSDP and +4.2 GiB of expert optimizer state.
**Net memory at the achieved configuration is roughly neutral, not the 3 GiB/tile
saving promised in §Goal.** Doc should retract the headline savings claim until a
batch-size sweep shows EP enables a configuration EP=1 cannot.

### 6. The EP=1 vs EP=4 throughput finding is buried

`project_ep_implementation.md` (April 11): EP=4 is **2.6× slower end-to-end** at
batch=1, grpo_samples=2 — strictly worse on every phase, including 24× slower opt
because expert AdamW runs on CPU. The doc's §"What's Next" item 3 cites a target of
"≤+10% step time", but the only comparable measurement so far shows +160%. This
should be in §Validation Status, not memory only.

### 7. Path forward — concrete experiments before more compute

Re-prioritized after the 2026-04-22 audit (hypothesis 1 closed, 3 closed,
`XPU_USM_ALLOC_SO` route closed):

1. **Try `use_reentrant=False` again** — highest leverage, no compute prerequisite.
   The v114 reason to use reentrant AC was a SIGSEGV in the **AllToAll** path, which
   is gone. Non-reentrant AC eliminates the entire "interleaved FWD-recompute and
   BWD" category that the leading hypothesis (#2) depends on. If this works, the EP
   path stops being autograd-fragile.
2. **Drop the NTPE all-gather collective** by deriving gather indices locally
   (`routed_input.shape[0]` + local `num_tokens_per_expert` are sufficient given
   the v110 interleaved routing). One fewer gloo collective per layer; removes one
   source of desync regardless of (1).
3. **Add NTPE counting + COLL labels to `_EP_OP_N`** (~5 LOC). The current
   `_EP_OP_N` skips NTPE-AG, so per-rank logs are off-by-one-per-layer in a way that
   makes the v153 forensics ambiguous. Pure diagnostic — only worth doing if (1)+(2)
   don't fix the desync and we need clearer logs to decide between hypothesis (2)
   and a still-unidentified candidate.
4. **Verify the `topk` → `argsort(stable=True)` and `histc` → `bincount` fixes in
   `gemma4/_moe.py`** (per `project_ep_implementation.md`) are still in place and
   produce identical routing across ranks for identical inputs. If routing diverged
   after some refactor, hypothesis (2) becomes a different problem entirely.
5. **If 1–4 fail**: implement option 4 (move dispatch outside autograd) — but only
   after the cheap experiments are exhausted. This is the largest change and is only
   worth attempting once the cheap fixes are ruled out.

**Do not** retry XCCL for AG/RS with the **arena** USM allocator — its slab
sub-allocation causes `zeMemGetIpcHandle` to return the slab base pointer,
not the sub-offset, triggering GPU page faults at 26B+ scale. The **caching**
allocator (`usm_caching_alloc.so`) does NOT sub-allocate — each pooled block
is a standalone `sycl::malloc_device` call with a valid IPC handle. The caching
allocator is production-ready at 3B and under validation at 32B.

### 8. What the doc should say but doesn't

- Where the actual EP wiring lives (recipe lines 1873–2189, not the §"Setup Sequence"
  meta-pseudocode).
- That `apply_ep_weight_sharding` is no longer called post-load; pre-FSDP2 meta-param
  shrinking is the current mechanism.
- That `_token_dispatch`/`_token_combine` carry **per-instance state** (`_ag_gather_idx`,
  `_ag_s_local`), but each MoE layer has its own `ExpertParallel` instance
  (`gemma4/_parallelism.py:49`), so there is no cross-layer aliasing. The relevant
  saved tensors for backward are captured by `_AllGatherRS.forward`'s `ctx.save_for_backward`,
  not from `self.*`, so within-layer FWD-recompute does not corrupt BWD inputs either.
  This rules out the per-instance-state hypothesis.
- That every EP collective currently bounces through CPU; performance numbers must be
  read with that caveat.
- The 2.6× regression at batch=1 (from project memory) — currently absent from §Validation
  Status entirely.

### 9. Audit summary (2026-04-22)

After the code audit, the open question is narrower than the doc previously suggested:

- **Closed**: local-rank-conditional collectives (none exist in the EP path);
  per-instance-state aliasing across layers (each layer has its own
  `ExpertParallel`); switching the USM allocator to dodge the IPC bug (the
  arena/caching paths have the same bug at this scale).
- **Open**: why local-index-1 ranks are *consistently* one labeled-op behind their
  EP-group peers at op #258. The leading explanation is reentrant-AC scheduling
  reacting to per-rank routing imbalance, but it is not proven.
- **Cheapest next step**: `use_reentrant=False` retry. It directly tests the leading
  hypothesis, requires no new infrastructure, and can be tried inside one debug-queue
  job. If non-reentrant AC desyncs identically, hypothesis (2) is wrong and we need
  to look at routing determinism (item 4 in §7) before anything else.

---

## v154 Result (2026-04-23) — non-reentrant AC exposes router non-determinism

Test: `use_reentrant=True → False` at recipe `_no_reentrant_ac_wrapper`
(`recipes/dev/grpo_full_finetune_distributed_xpu.py:157-165`).

**Outcome (job 8445872, hold_ep_fsdp2_v154.out):**
- The op-#259 RS-BWD desync from v153 is **gone**. EP collective lockstep held
  through op #182 across all 12 ranks.
- New failure at `(_c_loss / grad_scale).backward()`:
  ```
  [rank8]:  RuntimeError: Function ScatterAddBackward0 returned an invalid gradient
           at index 1 - got [4579, 2816] but expected shape compatible with [4578, 2816]
  [rank10]: RuntimeError: Function ScatterAddBackward0 returned an invalid gradient
           at index 1 - got [7505, 2816] but expected shape compatible with [7506, 2816]
  ```
  Off-by-one in opposite directions → one token migrated between ranks 8 and 10
  during AC recompute relative to the original FWD.

**Mechanism.** `ExpertParallel._token_dispatch` saves `_ag_gather_idx` /
`_ag_s_local` as **instance attributes** on the `ExpertParallel` object
(`torchtune/modules/moe/_parallelism.py:511, 578`). Non-reentrant AC re-runs
the FWD during BWD; the recompute overwrites those attributes with
freshly-derived values. If the recomputed router (`Gemma4MoERouter` —
sigmoid + `argsort(stable=True)`) produces even bitwise-different `scores`,
top-k assignments at tie boundaries can flip → `bincount` differs by ±1
across ranks within an EP group → `_ag_gather_idx.numel()` shifts → the
saved `routed_output` tensor and the recomputed `idx_exp` no longer match in
shape during `partial_out.scatter_add(0, idx_exp, routed_output)`. The
gradient for `scatter_add` then has the recomputed gather count, not the
original. Reentrant AC happened to mask this because reentrant recompute
runs as a sub-graph that re-derives both sides consistently within one frame.

**Stale comment.** Recipe lines 144–149 describe a `_fwd_step_counter` /
`_is_reuse` / saved-`ntpe_group` cache that was supposed to make AC recompute
reuse the original AllToAll counts. **That cache does not exist anywhere in
the current source** (`grep '_fwd_step_counter\|_is_reuse\|saved_ntpe' torchtune/`
returns nothing). Either it was removed during the v141 AllToAll → AllGather/RS
rewrite, or it never landed. The recipe comment should be deleted or rewritten.

**Hypothesis status.** v154 confirms half of the v153 conclusion: the op-#259
desync was tied to reentrant AC scheduling, not to the gloo/CXI stack or the
NTPE collective. The "rank with smallest routed batch falls one op behind"
story for **collective-ordering** desync looks right. The router-determinism
problem is orthogonal and was previously hidden because reentrant AC papered
over it.

### Decision: pause before more compute

Three plausible fixes; none is obviously cheapest, and the right pick depends
on whether we want to keep AC at all:

1. **Cache `_ag_gather_idx` and `_ag_s_local` per AC region.** Add a
   per-instance save/restore so the recompute reuses the values from the
   original FWD instead of regenerating them. Smallest diff, but every
   `ExpertParallel` instance now holds a tensor-sized side-channel through
   the BWD pass — easy to forget on multi-step training, and it does not fix
   the underlying router-determinism issue (it just ensures both sides agree
   on the *first* answer). ~30 lines in `_parallelism.py`.
2. **Mark router outputs as saved tensors.** Force AC to keep
   `(scores, selected_experts, num_tokens_per_expert)` from the original FWD
   instead of recomputing them. Fixes the determinism issue at its source but
   adds memory (router outputs are O(T·num_experts) per layer; for 30 layers,
   not free). Requires a custom `checkpoint_fn` that snapshots through the
   router boundary.
3. **Move EP dispatch/combine outside autograd.** The plan from §7 option 4 of
   "What's Next" — restructure dispatch/combine as side-effecting ops with
   manual gradient handling. Cleanest decoupling and reusable for any
   MoE + AC combination, but the largest diff and the only one that needs a
   real plan before implementation.

The v155–v157 sequence (drop NTPE AG, add NTPE labels, routing-determinism
probes) was designed assuming reentrant AC; it doesn't address the
ScatterAddBackward0 mismatch and would burn allocation chasing the wrong
target. Hold-node 8445872 was returned without further launches.

**Status.** v154 hypothesis on collective-ordering desync confirmed. Router-
recompute determinism is the next blocker. No more compute until we pick one
of fixes 1–3 above.

---

## Qwen3-30B-A3B Expert Parallelism — v1–v8 (2026-04-28 → 2026-04-30)

**Status (2026-04-30):** First `loss=` line landed on Qwen3 EP at v8i
(`EP=4/DP_replicate=6/DP_shard=4/fbs=1/G=1/NSTEPS=1` on 3-node hold), rc=0.
Both **G≥2** (real GRPO advantage variance) and **NSTEPS≥2** (multi-step
training) remain blocked by a unified architectural issue (v8m_b + v8n share
one root cause — see "v8 series unified blocker" subsection below). Production
RL continues on EP=1 (G=8, 54.8s/step). Full writeup:
`docs/reports/qwen3_ep_v8_3node_20260430.md`.



First end-to-end EP=4/DP=3 attempts on the new Qwen3MoE infrastructure.
Architecture: 2 nodes, 1 vLLM (3×TP=4) + 1 train (12 tiles EP=4/DP=3). Recipe
`recipes/dev/run_qwen3_30b_ep4_vllm_2node.sh` calls the standard GRPO recipe
with `qwen3_30b_a3b_grpo_ep4_xpu.yaml`. PBS jobs 8453156 (v1/v2) and 8453205
(v3-v7).

### Why this works where Gemma4 didn't (so far)

The Qwen3MoE codepath sidesteps every Gemma4 v141-v161 blocker:

| Gemma4 blocker | Qwen3 status |
|----------------|--------------|
| BWD desync at op #259 (autograd hook ordering) | Not seen — fwd lockstep across 12 ranks |
| ScatterAddBackward0 ±1 mismatch (router non-determinism in AC recompute) | Not seen — `Qwen3MoeTransformerLayer` puts MoE outside AC, so no router recompute |
| AsymmetricAG-BWD deadlock (v158) | Not reached — crashes before bwd starts |
| Reentrant AC tie-flip | N/A — `use_reentrant=False` from the start |

Qwen3 EP forward through ref/policy passes (no_grad, chunked) is reliable and
clean. **Failure mode is new and isolated to the train fwd.**

### v1-v7 result table

| Run | Knob | Furthest op (train fwd) | Crash | Notes |
|-----|------|-------------------------|-------|-------|
| v1 | baseline (gloo timeout 120s) | ~op #100 | gloo timeout | `XPU_USM_ALLOC_SO` leaked from held shell |
| v2 | gloo timeout 1800s + per-phase timing | op #427, layer ~13 | banned:1 PDE | coll= grew 19s→134s/layer monotonically |
| v3 | Tier 1: `CCL_ZE_CACHE=65536`, `unset XPU_USM_ALLOC_SO`, `gc:0.99` | op #425 | banned:1 PDE | coll= flat ~10s — gloo pressure fixed; PDE not |
| v4 | + `torch.xpu.synchronize()` after H2D in EP collectives | op #427 | banned:1 PDE | Async H2D use-after-free disproven |
| v5 | `forward_batch_size=4` (no chunking) | pass-2 OOM @ 60.83 GiB | clean Python OOM | Ranks 3/7/11 first → expert-load imbalance |
| v6 | `max_generated_tokens=128→64` (½ acts) | op #439, layer ~27 | banned:0 PDE | First time past v3's #425 — looked memory-bound |
| v7 | `max_gen=32 grpo_samples=4→2` (¼ acts vs v3) | op #253-equiv, layer ~30 | banned:1 PDE | 4× reduction → only +3 layers; **memory hypothesis disproven** |

### What we ruled out

1. **Gloo CPU pressure** (v3): the 134s coll= escalation was the fixable part —
   `CCL_ZE_CACHE=65536` + `unset XPU_USM_ALLOC_SO` + `gc:0.99` collapses it to
   flat 10s/layer. But the GPU PDE crash persists, so gloo pressure was a
   correlated symptom, not the cause.
2. **Async H2D use-after-free in EP collectives** (v4): adding
   `torch.xpu.synchronize()` after the gloo→XPU H2D copy in `_ep_all_gather` /
   `_ep_reduce_scatter` did not change the crash.
3. **No_grad → grad transition** (v5): turning off chunking gave a clean
   Python OOM in pass 2 (the policy fwd), not a banned:1 — so the
   `with torch.no_grad():` exit isn't the trigger.
4. **Pure activation memory pressure** (v7): a 4× activation reduction relative
   to v3 (max_gen 128→32 + grpo_samples 4→2) gained only ~3 layers vs v6's 2×
   reduction (layer 27→30). With proportional scaling we would have expected
   layer ~41. Diminishing returns rule out pure memory.
5. **Saved-tensor pinning of bounce-CPU buffers**: confirmed by reading
   `_AllGatherRS` and `_ReduceScatterAG` in `torchtune/modules/moe/_parallelism.py`
   — both save **only `ctx.group`**, no tensors. Backward recomputes via the
   inverse collective from `grad_output` only.

### What we know for sure

- Crash is always in the **train fwd** (ref + policy fwds run no_grad and
  succeed every time on the same EP code path).
- Crash address is in the L0 IPC range (`0xff01...`), not the standard USM
  device range (`0xff00...`).
- Crash position scales weakly with activation size — implying it's correlated
  with how much state autograd holds across EP collectives, not the activation
  bytes themselves.
- The recipe's own `device_empty_cache` is monkey-patched to no-op on XPU —
  not a confound here.

### Working hypothesis at end of v7 (REFUTED 2026-04-30 by v8g MEMPROBE)

L0 IPC handle pressure accumulates only when autograd builds its graph over EP
collectives. Each `_AllGatherRS.forward` / `_ReduceScatterAG.forward` registers
the gloo SHM IPC handles needed for the (eventual) gloo→XPU bounce; with 48
MoE layers × 2 EP collectives × 5 fwd passes per GRPO step that's ~480 IPC
handle registrations per training step. `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536`
helps but apparently isn't enough when autograd is also live.

**This hypothesis was wrong.** v8g instrumented every `_ep_all_gather` /
`_ep_reduce_scatter` ENTER/EXIT with `torch.xpu.mem_get_info()` and
`torch.xpu.memory_allocated()` on rank 0. Through 261 ops, `external = l0_used
- torch_alloc` stayed FLAT at ~3.4 GiB while `torch_alloc` walked linearly
from 15.6 → 60.4 GiB. The crash was **straightforward saved-activation OOM**;
the `banned:1 PDE` signature is just how XPU surfaces an L0 OOM during a CCS
write that overruns the allocation map.

## v8 series — first `loss=` line + unified architectural blocker (2026-04-30)

3-node hold (1 dedicated vLLM + 2 train), `EP=4 / DP_replicate=6 / DP_shard=4`,
world=24. EP groups all intra-node. Launcher
`recipes/dev/run_qwen3_30b_ep4_vllm_3node.sh`, per-run wrapper
`experiments/ep_parallelism/hold_qwen3_ep_v8.sh`. PBS jobs 8459180 + 8459706.

| Run | Knob | Result | Notes |
|-----|------|--------|-------|
| v8a | v7 settings on 3-node | crash op #261 banned:1 | 3-node DP=6 same regime as v7 — **refutes "cross-node TCP-gloo eliminates SHM IPC" hypothesis** (all EP groups intra-node here) |
| v8g | + `_ep_mem_probe()` per EP-OP | crash op #261 (instrumented) | **DEFINITIVE diagnosis**: external FLAT at ~3.4 GiB, all 45 GiB growth in `torch_alloc`. PDE was activation OOM. |
| v8h | `forward_batch_size=1` | full fwd + chunk-1 bwd OK; OOM in chunk-2 fwd op #624 | chunk-1 grads stay live (FSDP2 unsharded) → chunk-2 fwd has no budget |
| v8i | `fbs=1 grpo_samples=1` | **first `loss=` line**, rc=0 | 33.7s/step. loss=nan (G=1 → zero advantage variance), kl_loss=0.000984, ratios=1.000, resp_len=31 |
| v8j | `fbs=1 G=1 max_gen=64` | step 1 rc=0 | 44.1s/step (+33% vs mg=32). loss=nan (sparse reward at G=1) |
| v8k | + `fsdp_cpu_offload=true G=2 fbs=1` | DNF (init >30 min) | FSDP2 cpu_offload H2D too slow at 30B. Cleanup left orphans → contaminated nodes |
| v8l, v8m | NSTEPS=5 / G=2 mg=16 on contaminated nodes | hung between gen and `chunk[0:1] fwd` | held-node contamination; mitigation = switch to fresh PBS job |
| v8m_b | `G=2 fbs=1 mg=16` on fresh nodes | crash in `chunk[1:2] fwd`, banned:1 | **chunk[0:1] PRE-BWD = 50.08 GiB at mg=16 vs 50.85 GiB at mg=32 — halving max_gen reduced acts by <1 GiB.** Activations dominated by fixed buffers (KV cache, prompts, ref_logprobs), NOT response tokens. |
| v8n | `NSTEPS=2 G=1 fbs=1 mg=32` on fresh nodes | step 0 clean; step 1 banned:1 in `chunk[0:1] fwd` | **PRE-STEP 1 alloc = 34.72 GiB** vs PRE-STEP 0 = 14.06 GiB. `optimizer.zero_grad(set_to_none=True)` does NOT release FSDP2's internal unsharded grad buffers in EP mode. |

### v8 series unified blocker (v8m_b + v8n share root cause)

v59's `reduce_grads=False` (`recipes/dev/grpo_full_finetune_distributed_xpu.py:1440-1464`)
forces FSDP2 to skip `post_backward` reduce_scatter on every FSDPParamGroup in
EP mode. The full unsharded grad pool (~18.5 GiB at DP_shard=4) stays resident:

- **between chunks within a step** → blocks G≥2 (v8m_b: chunk-2 fwd has to fit
  activations on top of chunk-1's unsharded grads)
- **between steps** → blocks NSTEPS≥2 (v8n: PRE-STEP 1 already at 34.72 GiB
  before any new activations are saved; `optimizer.zero_grad(set_to_none=True)`
  at lines 3047/3322 doesn't reach FSDP2's internal `FSDPParamGroup` grad
  buffers, only `optimizer.param_groups[*]['params']`)

Halving `max_gen` is not a viable tuning fix because activations are dominated
by fixed-cost buffers (KV cache for prompt+ref, ref_logprobs storage, prompt
sequence buffers), not response-token activations.

The current path's grad sync is a single deferred XCCL all_reduce on the
dp_replicate group AFTER all chunks
(`grpo_full_finetune_distributed_xpu.py:2945-2954`).

### Real fix (not implemented this session)

Add explicit `fully_shard`-aware reduce_scatter immediately after each chunk's
bwd (~lines 2945-2954) AND after the optimizer step (~lines 3047, 3322), then
sequence with the existing deferred dp_replicate all_reduce so grads aren't
double-reduced. This is the one code change that unblocks BOTH G≥2 AND
NSTEPS≥2.

### Diagnostic rule for future XPU `banned:0/1 PDE` crashes

For any future XPU `banned:0/1 PDE` crash with addresses in `0xff0[0-5]...`:

1. FIRST add the `_ep_mem_probe`-style trace
   (`torchtune/modules/moe/_parallelism.py:108-135`, opt-in via rank-0 gate) to
   localize whether `torch_alloc` is climbing or `external` is climbing.
2. If `external` is FLAT, it's an activation OOM and the fix is per-tile budget
   (`forward_batch_size`, `grpo_samples`, `max_generated_tokens`,
   activation checkpointing) — NOT env vars or PG transport changes.
3. The PDE signature is a red herring; XPU surfaces L0 OOM as a CCS-write
   fault that overruns the allocation map.

Pluggable allocator ON (`XPU_USM_ALLOC_SO` set) blinds these probes
(`mem_get_info` / `memory_stats` are no-ops). The 3-node Qwen3 EP launcher
leaves `XPU_USM_ALLOC_SO` unset on train ranks (vLLM rank pops it explicitly
per `grpo_full_finetune_distributed_xpu.py:71-73`), so probes are valid.

### Held-node hygiene rule (from v8k aftermath)

If a heavy training launch — especially `fsdp_cpu_offload=true` or any path
that loads 30B params per rank — fails partway through init and leaves orphans,
the held nodes are likely contaminated for further EP runs **even after
`pkill -9`**. Symptom: subsequent training hangs between gen and `grpo_step`
with no error. Mitigation: switch to a fresh PBS job's nodes; orphan-free
pkill alone is insufficient.

### Files of record

- Launcher (per-run, v1-v7): `experiments/ep_parallelism/hold_qwen3_ep_v{1-7}.sh`
- Launcher (per-run, v8): `experiments/ep_parallelism/hold_qwen3_ep_v8.sh`
  (single wrapper, runs v8a/g/h/i/j/k/l/m/m_b/n via env vars)
- Output logs: `experiments/ep_parallelism/hold_qwen3_ep_v{1-7,8,8a,8g,8h,8i,8j,8k,8l,8m,8m_b,8n}.out`
- Common training launchers:
  `recipes/dev/run_qwen3_30b_ep4_vllm_2node.sh` (v1-v7),
  `recipes/dev/run_qwen3_30b_ep4_vllm_3node.sh` (v8 series)
- vLLM server launcher: `recipes/dev/run_qwen3_30b_vllm_server.sh`
- Config: `recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep4_xpu.yaml`
- EP collective definitions: `torchtune/modules/moe/_parallelism.py:193-227`
  (`_AllGatherRS`, `_ReduceScatterAG`)
- EP memory probe: `torchtune/modules/moe/_parallelism.py:108-135`
  (`_ep_mem_probe()`, opt-in via rank-0 gate; landed in v8g)
- EP plan: `torchtune/models/qwen3_moe/_parallelism.py` (`qwen3_moe_ep_plan`)
- Reduce-grads suppression site (root of the v8 unified blocker):
  `recipes/dev/grpo_full_finetune_distributed_xpu.py:1440-1464` (v59);
  deferred dp_replicate all_reduce: `:2945-2954`;
  optimizer.zero_grad call sites that don't reach FSDP2 grads: `:3047, 3322`
- Synthesis report: `docs/reports/qwen3_ep_v8_3node_20260430.md`
- Memory entries: `project_qwen3_ep_v1_v2.md`, `project_qwen3_ep_v3_v7.md`,
  `project_qwen3_ep_v8_3node.md`
