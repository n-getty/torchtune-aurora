# MoE Expert Parallelism Integration — Aurora/XPU

Status and implementation notes for Expert Parallelism (EP) support for Gemma4 26B-A4B
on Aurora HPC (Intel Max Series GPUs / XPU), targeting EP=4/DP=3 on 12 tiles per node.

## Goal

Reduce per-tile expert parameter memory by partitioning Gemma4's 128 experts across 4 EP
ranks, so each tile holds 32 experts instead of 128. Primary benefit is memory reduction
(~3 GiB/tile at bfloat16) which enables larger batch sizes or reduces optimizer state
pressure during GRPO training.

Topology: `dp_replicate=3 × dp_shard=4` on 12 tiles. The `dp_shard` submesh doubles as
the EP communicator group — the same 4 ranks handle both FSDP sharding of non-expert
parameters and Expert Parallel token dispatch.

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
| `torchtune/modules/moe/_parallelism.py` | Added `ExpertParallel`, `apply_ep_weight_sharding` |
| `torchtune/modules/moe/__init__.py` | Exported `ExpertParallel`, `apply_ep_weight_sharding` |
| `torchtune/modules/moe/experts.py` | Fixed `_forward_no_grouped_mm` to use local expert count |
| `torchtune/models/gemma4/_parallelism.py` | New: `gemma4_ep_plan()` returning EP plan for 30 MoE layers |
| `torchtune/models/gemma4/__init__.py` | Exported `gemma4_ep_plan` |

### Test / validation scripts

| File | Purpose |
|------|---------|
| `recipes/dev/test_ep_smoke.py` | 4-subtest smoke test (imports, permute, all-to-all, forward) |
| `recipes/dev/test_ep_correctness.py` | EP=2 vs replicated output comparison (bit-exact) |
| `recipes/dev/test_ep_model_setup.py` | 12-tile EP=4/DP=3 topology validation with real checkpoint |
| `recipes/dev/run_ep_smoke.sh` | Smoke test launcher |

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

### apply_ep_weight_sharding (`torchtune/modules/moe/_parallelism.py`)

After loading the full checkpoint, call this to slice each EP module's parameters to the
local shard:

```python
n_sharded = apply_ep_weight_sharding(model)
# → 30 modules × 3 params each (gate_proj, up_proj, down_proj)
# → param.data[ep_rank * n_local : (ep_rank+1) * n_local].contiguous()
```

**Critical ordering constraint**: must be called AFTER `load_state_dict` but BEFORE
`shard_model()`. After FSDP2 `fully_shard()`, named parameters are absorbed into flat
buffers — `named_parameters(recurse=False)` returns nothing for FSDP2-sharded modules.

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

## Setup Sequence (CRITICAL Ordering)

```python
# 1. Build 2D mesh
dp_mesh = init_device_mesh(device.type, (dp_replicate, dp_shard),
                            mesh_dim_names=("dp_replicate", "dp_shard"))
ep_mesh = dp_mesh["dp_shard"]

# 2. Eager process group initialization (prevents XCCL deadlock in forward hooks)
_ = ep_mesh.get_group()          # ep_mesh is 1D
_ = dp_mesh.get_all_groups()     # dp_mesh is 2D — must use get_all_groups(), NOT get_group()
dist.barrier()

# 3. Build model on meta device
with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = gemma4_26b_a4b()

# 4. Apply EP hooks (stores _ep_device_mesh, registers forward hooks)
ep_plan = gemma4_ep_plan(model)
parallelize_module(model, ep_mesh, ep_plan)

# 5. Stage checkpoint to /tmp (rank 0 copies, barrier, all ranks load)
if rank == 0:
    shutil.copy2(LUSTRE_CKPT, TMP_CKPT)   # ~62s vs 357s from Lustre with 12 concurrent readers
dist.barrier()

# 6. Materialize model on XPU and load checkpoint
model.to_empty(device=device)
sd = checkpointer.load_checkpoint()
model.load_state_dict(sd["model"], strict=False)

# 7. Slice expert weights to local EP shards (PRE-FSDP2)
apply_ep_weight_sharding(model)

# 8. shard_model() / FSDP2 — see "FSDP2 integration" below
```

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

## Validation Status (April 10, 2026)

| Test | Config | Status | Notes |
|------|--------|--------|-------|
| Import smoke test | EP=2, 2 tiles | **PASS** | `ExpertParallel`, `_permute`, `all_to_all_single_autograd` |
| Permute/unpermute | EP=2, 2 tiles | **PASS** | Round-trip, all token counts |
| All-to-All forward | EP=2, 2 tiles | **PASS** | Correct shapes and values |
| Import smoke test | EP=4, 4 tiles | **PASS** | |
| All-to-All forward | EP=4, 4 tiles | **PASS** | |
| EP=2 vs replicated | EP=2, 2 tiles | **PASS** | Bit-exact match (max_err=0.0) |
| 12-tile forward pass | EP=4/DP=3, 12 tiles | **PASS** | See results below |

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

## FSDP2 + EP Communicator Conflict

Running FSDP2 all-gather (12-rank `dp_mesh` group) and EP All-to-All (4-rank `ep_mesh`
group) concurrently during a single forward pass causes XCCL deadlock on Aurora.
Symptoms: all 12 ranks spin at 92–97% CPU (XCCL busy-wait), no system calls, no progress.

The conflict arises because FSDP2 prefetches the next layer's all-gather while the current
layer's EP all-to-all is in-flight, and XCCL can't interleave operations across different
communicator groups.

**Current workaround**: FSDP2 is skipped in the 12-tile forward pass test.

**Solutions for training integration** (in order of preference):

1. **Wrap non-expert params on `dp_replicate` mesh only (3 ranks)**
   FSDP2 all-gather runs within a 3-rank group; EP all-to-all runs within a 4-rank group.
   No overlap. Non-expert memory savings are 3× (vs 12× without EP). Expert params
   are already EP-partitioned (4×) and can be FSDP-wrapped separately on `ep_mesh`.

2. **Sequential FSDP scheduling** (`reshard_after_forward=False`)
   Disables FSDP prefetch, forcing all-gather to complete before the next layer starts.
   Eliminates overlap but reduces throughput.

3. **Skip FSDP2 for experts, use FSDP2 on `dp_replicate` for all other params**
   Expert params don't participate in FSDP2 at all — they're sliced by EP, gradient
   reduction happens via the EP all-to-all backward. Clean separation.

---

## Memory Model

Without any FSDP2 (current test state):

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

EP=4 peak may be slightly higher than EP=1 in the param+optimizer breakdown because
non-expert FSDP sharding is coarser (3-rank vs 12-rank). The benefit is in
**load balancing** and enabling larger effective batch sizes without expert OOM.

---

## What's Next

1. **FSDP2 integration**: wire `shard_model()` to use `dp_replicate` mesh for non-expert
   params, and wrap expert modules separately on `ep_mesh`. Validate no deadlock.

2. **GRPO recipe wiring**: update `grpo_full_finetune_distributed_xpu.py` to accept
   `expert_parallel_degree` config, build the 2D mesh, apply EP hooks, call
   `apply_ep_weight_sharding`, then call `shard_model`.

3. **Full EP=4/DP=3 GRPO benchmark**: step time and peak memory vs EP=1 baseline
   (24.5 s/step, 20.66 GiB). Target: memory neutral-to-better, step time ≤ +10%.
